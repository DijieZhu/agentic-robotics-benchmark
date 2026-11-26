"""The check of a submitted description of the OpenArm pair: one MJCF file and its calibration. PUBLIC.

PUBLIC: this module ships with the public runner of a submitted model. It imports the standard library, MuJoCo and the
public modules `base` and `state` (which read the maker's vendored files and params_public.json); nothing of a drawn
unit. Its lists and caps are read from params_public.json (`mjcf_policy`).

What a description is: ONE MJCF file holding the pair on the maker's body and joint names (enactic/openarm_mujoco, folder
v2, commit ed2046f30981398fc610688f08beac0ebd08f972, scene pedestal/pedestal.xml), with two tool sites and sixteen
`general` actuators, and a calibration file
    {"format": "openarm-pair-calibration/1", "joints": {<arm joint>: {"sign": +1 | -1, "zero_offset_rad": z}, ...}}
for the fourteen arm joints (bus angle b = sign x (q - z), q the model's joint coordinate).

The check runs in this order and stops at the first stage that fails (each failure is a (check, reason) pair):
 1. files: the byte caps of both files;
 2. text: well-formed XML (no document type or entity declaration), none of the forbidden elements anywhere, a file
    attribute only on a <mesh> whose file base name is one of the maker's given mesh files, no inline mesh data; the
    calibration is well-formed JSON. A <frame> element is accepted: it only places what it holds, the compiler
    removes it, and the form below is read on the compiled model. An <asset><model> element is refused (it names
    another model file, the source of an <attach>). A <size> element may reserve at most 16 keyframes and no user
    data, and leaves its memory, njmax, nconmax and nstack attributes absent or -1 (MuJoCo's default): the size
    element plays no part in a run (the rebuild replaces it), but the file is compiled as written first, and the
    compiler allocates what the element asks for;
 3. compile: MuJoCo 3.14.0 compiles the file with zero warnings, the file's meshdir, assetdir and texturedir replaced
    by the given assets folder and every mesh file named by its base name in that folder;
 4. form, on the COMPILED model (so that compiler settings and default classes act as MuJoCo resolves them):
    - bodies: the twenty arm bodies by name in the maker's parent-child order (mount_* bodies skipped), no other body;
      no body with gravity compensation, no mocap body;
    - mounts: mount_<side>_<k> (k = 1..4) leaves under an arm body of the same side, no joint, at most one primitive
      geom, at most four per side;
    - joints: exactly the eighteen hinges in the maker's order, each on its body; stiffness 0 (every term), spring
      reference equal to the reference, reference 0, damping by its linear term only (higher terms 0), no actuator
      gravity compensation;
    - equalities: exactly the maker's two finger equalities, each a joint equality between that side's two finger
      joints;
    - actuators: exactly the sixteen by the maker's names and order, each with a joint transmission on its joint, gear
      [1 0 0 0 0 0], no activation dynamics, a fixed gain whose terms after the first are 0, an affine bias whose terms
      after the first are 0, and no actuator damping, armature or delay;
    - tool sites: left_tool on openarm_left_ee_base_link, right_tool on openarm_right_ee_base_link;
    - every number that is read (body frames and inertials, joint frames, ranges, armature, damping, friction, gains,
      biases, site frames) finite;
 5. calibration: the fourteen arm joints present, sign +1 or -1, finite offsets with |offset| <= pi (other keys
    ignored);
 6. caps, on the compiled model: counts of bodies and geoms; every body's mass inside the cap with a physical inertia
    (positive principal inertias, any two summing to at least the third); every joint limited with lower < upper,
    both inside +-2 pi; armature, damping and friction loss inside their caps; k and b inside their caps.

A file that passes is a description that the public runner (frommodel.py) rebuilds and runs; nothing in it is
executed.
"""
import json
import math
import os
import re
import warnings as _pywarnings
import xml.etree.ElementTree as ET

import mujoco

from . import base
from .state import public_params

POLICY_FORMAT = "openarm_unit/ModelPolicy/1"
CALIBRATION_FORMAT = "openarm-pair-calibration/1"
STAGES = ("files", "text", "compile", "form", "calibration", "caps")
REPLACED_COMPILER_ATTRIBUTES = ("meshdir", "assetdir", "texturedir")
TOOL_SITES = {"left": ("left_tool", "openarm_left_ee_base_link"), "right": ("right_tool", "openarm_right_ee_base_link")}
EQUALITY_NAMES = ("openarm_left_ee_finger_joint_mimic", "openarm_right_ee_finger_joint_mimic")
MOUNT_NAME_RE = base.MOUNT_NAME
PRIMITIVE_GEOMS = tuple(sorted(base.PRIMITIVE_SIZES))
_DECL_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
# What a <size> element may ask the compiler for (the file is compiled as written before the rebuild replaces the
# element): at most this many keyframes, no user data (every count at most 0; -1, the compiler's default for the
# nuser_<kind> counts, reserves none; the compiler refuses nuserdata=-1), and the memory and buffer sizes left at the
# compiler's default (absent or -1). A keyframe or user-data count that is not a whole number is left to the
# compiler, which refuses it.
SIZE_MAX_KEYFRAMES = 16
SIZE_MAX_USER_DATA = 0
SIZE_USER_DATA_ATTRIBUTES = ("nuserdata", "nuser_body", "nuser_jnt", "nuser_geom", "nuser_site", "nuser_cam",
                             "nuser_tendon", "nuser_actuator", "nuser_sensor")
SIZE_DEFAULT_ONLY_ATTRIBUTES = ("memory", "njmax", "nconmax", "nstack")
_WHOLE_NUMBER_RE = re.compile(r"^\s*[+-]?[0-9]+\s*$")      # what the compiler reads as a whole number


class PolicyError(ValueError):
    """A description that does not pass the check; `result` holds every stage's verdict."""

    def __init__(self, result):
        self.result = result
        super().__init__(result.first_failure() or "the description does not pass the check")


def policy():
    """params_public.json `mjcf_policy`: the lists (keep, accept and ignore, forbidden, refused on the compiled
    model) and the caps, as plain values."""
    p = public_params()["mjcf_policy"]
    return {k: v["value"] for k, v in p.items()}


def caps():
    return dict(policy()["caps"])


def forbidden_elements():
    return tuple(policy()["forbidden_elements"])


_GIVEN = None


def given_mesh_files():
    """{file base name: path relative to the maker's v2/assets} of every mesh file the maker's pedestal scene names
    (the given set). Base names are unique in the set."""
    global _GIVEN
    if _GIVEN is None:
        spec = base.maker_spec()
        out = {}
        for m in spec.meshes:
            f = str(m.file).replace("\\", "/")
            if not f:
                continue
            bn = f.rsplit("/", 1)[-1]
            if bn in out and out[bn] != f:
                raise base.BaseIntegrityError(f"two given mesh files share the base name {bn}")
            out[bn] = f
        _GIVEN = out
    return dict(_GIVEN)


def given_assets_dir():
    """The vendored copy of the maker's v2/assets (verified against its manifest)."""
    base.verify_base()
    return str((base.BASE_DIR / base.MAKER_FOLDER / "assets").resolve())


def arm_body_tree():
    """((body, parent), ...) of the twenty arm bodies in the maker's compiled order ('world' = the world body)."""
    out = []
    for s in base.SIDES:
        b = base.ARM_BODIES[s]
        out.append((b[0], "world"))
        for k in range(1, 8):                    # link1 .. link6, ee_base_link: each on the one before
            out.append((b[k], b[k - 1]))
        out.append((b[8], b[7]))                 # both fingers on ee_base_link
        out.append((b[9], b[7]))
    return tuple(out)


def hinge_bodies():
    """((joint, body), ...) of the eighteen hinges in the maker's compiled order."""
    out = []
    for s in base.SIDES:
        b = base.ARM_BODIES[s]
        for k in range(1, 8):
            out.append((f"openarm_{s}_joint{k}", b[k]))
        out.append((f"openarm_{s}_finger_joint1", b[8]))
        out.append((f"openarm_{s}_finger_joint2", b[9]))
    return tuple(out)


# ------------------------------------------------------------------------------------------------------ the result
class PolicyResult:
    """Per stage: 'pass', 'fail' or 'not reached', with its failures [(check, reason)]. After a pass: `model` (the
    compiled file), `text` (the file as compiled: mesh folder replaced), `calibration` {arm joint: (sign, offset)},
    `xml_bytes`, `cal_bytes`."""

    def __init__(self):
        self.stages = {s: {"verdict": "not reached", "failures": []} for s in STAGES}
        self.model = None
        self.text = None
        self.calibration = None
        self.xml_bytes = None
        self.cal_bytes = None

    def set(self, stage, failures):
        self.stages[stage] = {"verdict": "fail" if failures else "pass", "failures": [list(f) for f in failures]}
        return not failures

    @property
    def passed(self):
        return all(v["verdict"] == "pass" for v in self.stages.values())

    def failures(self):
        return [(s, c, r) for s in STAGES for c, r in self.stages[s]["failures"]]

    def first_failure(self):
        f = self.failures()
        return None if not f else f"{f[0][0]}: {f[0][1]}: {f[0][2]}"

    def summary(self):
        return {"format": POLICY_FORMAT, "passed": self.passed,
                "stages": {s: {"verdict": v["verdict"], "failures": [list(x) for x in v["failures"]]}
                           for s, v in self.stages.items()}}


# ------------------------------------------------------------------------------------------------------- 1. files
def file_failures(xml_bytes, cal_bytes):
    c = caps()
    out = []
    if len(xml_bytes) > int(c["robot_xml_max_bytes"]):
        out.append(("files", f"robot.xml has {len(xml_bytes)} bytes; at most {int(c['robot_xml_max_bytes'])}"))
    if len(cal_bytes) > int(c["calibration_max_bytes"]):
        out.append(("files", f"calibration.json has {len(cal_bytes)} bytes; at most {int(c['calibration_max_bytes'])}"))
    return out


# -------------------------------------------------------------------------------------------------------- 2. text
def _basename(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def size_failures(el):
    """The failures of one <size> element: one per attribute that breaks a rule, in the element's attribute order."""
    out = []
    for a, v in el.attrib.items():
        whole = _WHOLE_NUMBER_RE.match(v) is not None
        if a == "nkey":
            if whole and int(v) > SIZE_MAX_KEYFRAMES:
                out.append(("size reservation", f"<size> reserves nkey={v.strip()!r}; at most {SIZE_MAX_KEYFRAMES} "
                                                "keyframes may be reserved"))
        elif a in SIZE_USER_DATA_ATTRIBUTES:
            if whole and int(v) > SIZE_MAX_USER_DATA:
                out.append(("size reservation", f"<size> reserves {a}={v.strip()!r}; no user data may be reserved"))
        elif a in SIZE_DEFAULT_ONLY_ATTRIBUTES:
            if not (whole and int(v) == -1):
                out.append(("size memory", f"<size> sets {a}={v.strip()!r}; memory, njmax, nconmax and nstack must "
                                           "be absent or -1"))
    return out


def text_failures(xml_text):
    """The text check of robot.xml. Returns (failures, root or None)."""
    if _DECL_RE.search(xml_text):
        return [("declarations", "a document type or entity declaration is not accepted")], None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return [("well-formed", f"robot.xml is not well-formed XML: {e}")], None
    if root.tag != "mujoco":
        return [("well-formed", f"the root element is <{root.tag}>, not <mujoco>")], root
    forbidden = set(forbidden_elements())
    given = given_mesh_files()
    out = []
    for el in root.iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in forbidden:
            out.append(("forbidden element", f"<{tag}>"))
            continue
        fa = [a for a in el.attrib if a.startswith("file")]
        if fa and tag != "mesh":
            out.append(("file reference", f"<{tag}> carries {fa[0]}=...; only a <mesh> of the given set may name a file"))
            continue
        if tag == "mesh":
            name = el.attrib.get("name", "?")
            extra = [a for a in fa if a != "file"]
            if extra:
                out.append(("file reference", f"mesh {name!r} has a file attribute {extra[0]!r}"))
            elif "file" not in el.attrib:
                out.append(("mesh data", f"mesh {name!r} names no file (inline or built-in mesh data is not accepted)"))
            elif _basename(el.attrib["file"]) not in given:
                out.append(("file reference", f"mesh file {_basename(el.attrib['file'])!r} is not a given mesh file"))
            elif any(a in el.attrib for a in ("vertex", "face", "normal", "texcoord", "builtin", "params")):
                out.append(("mesh data", f"mesh {name!r} carries inline mesh data"))
        if tag == "size":
            out.extend(size_failures(el))
    return out, root


def calibration_parse(cal_bytes):
    try:
        return json.loads(cal_bytes.decode("utf-8")), None
    except (UnicodeDecodeError, ValueError) as e:
        return None, f"calibration.json is not well-formed JSON: {e}"


# ----------------------------------------------------------------------------------------------------- 3. compile
def grader_text(root, assets_dir=None):
    """The file as it is compiled: every compiler's meshdir / assetdir / texturedir removed, meshdir set to the given
    assets folder, every mesh file named by its path in that folder."""
    assets_dir = os.path.abspath(assets_dir or given_assets_dir())
    given = given_mesh_files()
    root = ET.fromstring(ET.tostring(root, encoding="unicode"))
    comps = root.findall("compiler")
    if not comps:
        comp = ET.Element("compiler")
        root.insert(0, comp)
        comps = [comp]
    for comp in comps:
        for a in REPLACED_COMPILER_ATTRIBUTES:
            comp.attrib.pop(a, None)
        comp.set("meshdir", assets_dir)
    for el in root.iter("mesh"):
        if "file" in el.attrib:
            el.set("file", given[_basename(el.attrib["file"])])
    return ET.tostring(root, encoding="unicode")


def compile_spec(text):
    """(spec, model, warnings, error): MuJoCo's spec of the text and its compiled model, every warning captured."""
    msgs = []
    old = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda m: msgs.append(str(m)))
    try:
        with _pywarnings.catch_warnings(record=True) as w:
            _pywarnings.simplefilter("always")
            try:
                spec = mujoco.MjSpec.from_string(text)
                m = spec.compile()
            except Exception as e:  # noqa: BLE001  (MuJoCo raises ValueError and its own errors)
                msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                return None, None, msgs + [str(x.message) for x in w], msg
        msgs += [str(x.message) for x in w]
    finally:
        mujoco.set_mju_user_warning(old)
    return spec, m, msgs, None


# -------------------------------------------------------------------------------------------------------- 4. form
def _nm(m, t, i):
    return mujoco.mj_id2name(m, t, i) or ""


def _side_of(name):
    for s in base.SIDES:
        if name.startswith(f"openarm_{s}_") or name.startswith(f"mount_{s}_"):
            return s
    return None


def _all_zero(a):
    return all(float(x) == 0.0 for x in a.reshape(-1))


def _all_finite(a):
    return all(math.isfinite(float(x)) for x in a.reshape(-1))


def form_failures(m):
    """The form on the compiled model: a list of (check, reason)."""
    O = mujoco.mjtObj
    out = []
    bodies = [_nm(m, O.mjOBJ_BODY, i) for i in range(m.nbody)]
    tree = arm_body_tree()
    names = tuple(b for b, _ in tree)
    # bodies
    arm = [(i, b) for i, b in enumerate(bodies) if i > 0 and not MOUNT_NAME_RE.match(b)]
    got = tuple((b, bodies[m.body_parentid[i]] if m.body_parentid[i] > 0 else "world") for i, b in arm)
    if got != tree:
        extra = [b for _, b in arm if b not in names]
        missing = [b for b in names if b not in bodies]
        if extra:
            out.append(("other bodies", f"body {extra[0]!r} is neither an arm body nor a mount_<side>_<k>"))
        elif missing:
            out.append(("bodies", f"body {missing[0]!r} is missing"))
        else:
            for (b, p), (eb, ep) in zip(got, tree):
                if (b, p) != (eb, ep):
                    out.append(("bodies", f"expected {eb!r} on {ep!r} at this place of the maker's order, found "
                                          f"{b!r} on {p!r}"))
                    break
    for i in range(1, m.nbody):
        if float(m.body_gravcomp[i]) != 0.0:
            out.append(("gravity compensation", f"body {bodies[i]!r} has gravcomp {float(m.body_gravcomp[i])!r}"))
        if int(m.body_mocapid[i]) >= 0:
            out.append(("mocap", f"body {bodies[i]!r} is a mocap body"))
    # mounts
    per_side = {s: 0 for s in base.SIDES}
    for i, b in enumerate(bodies):
        mm = MOUNT_NAME_RE.match(b) if i > 0 else None
        if not mm:
            if i > 0 and b.startswith("mount_"):
                out.append(("mounts", f"{b!r} is not of the form mount_<side>_<k> with k = 1..4"))
            continue
        side = mm.group(1)
        per_side[side] += 1
        parent = bodies[m.body_parentid[i]]
        if parent not in names or _side_of(parent) != side:
            out.append(("mounts", f"{b} is not under an arm body of its side (parent {parent!r})"))
        if int(m.body_jntnum[i]) != 0:
            out.append(("mounts", f"{b} has a joint"))
        if int(m.body_geomnum[i]) > 1:
            out.append(("mounts", f"{b} has {int(m.body_geomnum[i])} geoms; at most 1"))
        for g in range(int(m.body_geomadr[i]), int(m.body_geomadr[i]) + int(m.body_geomnum[i])):
            gt = mujoco.mjtGeom(int(m.geom_type[g])).name.replace("mjGEOM_", "").lower()
            if gt not in PRIMITIVE_GEOMS:
                out.append(("mounts", f"{b} carries a {gt} geom; only a primitive is accepted"))
        if any(int(m.body_parentid[k]) == i for k in range(1, m.nbody) if k != i):
            out.append(("mounts", f"{b} is not a leaf"))
    cap = int(caps()["max_mounts_per_side"])
    for s, n in per_side.items():
        if n > cap:
            out.append(("mounts", f"{n} mounts on the {s} side; at most {cap}"))
    # joints
    want = hinge_bodies()
    joints = tuple((_nm(m, O.mjOBJ_JOINT, j), _nm(m, O.mjOBJ_BODY, int(m.jnt_bodyid[j]))) for j in range(m.njnt))
    if joints != want:
        if len(joints) != len(want):
            out.append(("joints", f"{len(joints)} joints; exactly the {len(want)} hinges of the maker are accepted"))
        else:
            for (j, b), (ej, eb) in zip(joints, want):
                if (j, b) != (ej, eb):
                    out.append(("joints", f"expected joint {ej!r} on {eb!r}, found {j!r} on {b!r}"))
                    break
    for j in range(m.njnt):
        jn = joints[j][0]
        if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            out.append(("joints", f"{jn} is not a hinge"))
            continue
        a, d = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        if float(m.jnt_stiffness[j]) != 0.0 or not _all_zero(m.jnt_stiffnesspoly[j]):
            out.append(("joint stiffness", f"{jn} has a stiffness term; every term must be 0"))
        if float(m.qpos_spring[a]) != float(m.qpos0[a]):
            out.append(("spring reference", f"{jn} has springref {float(m.qpos_spring[a])!r}, its reference is "
                                            f"{float(m.qpos0[a])!r}"))
        if float(m.qpos0[a]) != 0.0:
            out.append(("joint ref", f"{jn} has ref {float(m.qpos0[a])!r}; it must be 0"))
        if not _all_zero(m.dof_dampingpoly[d]):
            out.append(("joint damping", f"{jn} has a damping term beyond the linear one"))
        if int(m.jnt_actgravcomp[j]) != 0:
            out.append(("gravity compensation", f"{jn} has actuatorgravcomp"))
    # equalities
    eqs = [_nm(m, O.mjOBJ_EQUALITY, e) for e in range(m.neq)]
    if sorted(eqs) != sorted(EQUALITY_NAMES):
        out.append(("equalities", f"equalities {eqs}; exactly {list(EQUALITY_NAMES)} are accepted"))
    else:
        for e in range(m.neq):
            side = "left" if "_left_" in eqs[e] else "right"
            pair = {mujoco.mj_name2id(m, O.mjOBJ_JOINT, f"openarm_{side}_finger_joint{k}") for k in (1, 2)}
            if (int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT)
                    or {int(m.eq_obj1id[e]), int(m.eq_obj2id[e])} != pair):
                out.append(("equalities", f"{eqs[e]} is not a joint equality between the {side} finger joints"))
    # actuators
    acts = tuple((_nm(m, O.mjOBJ_ACTUATOR, a), _nm(m, O.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
                  if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) else "") for a in range(m.nu))
    if acts != tuple(base.ACTUATORS):
        if len(acts) != len(base.ACTUATORS):
            out.append(("actuators", f"{len(acts)} actuators; exactly the sixteen of the maker are accepted"))
        else:
            for (a, j), (ea, ej) in zip(acts, base.ACTUATORS):
                if (a, j) != (ea, ej):
                    out.append(("actuators", f"expected {ea!r} with a joint transmission on {ej!r}, found {a!r} on "
                                             f"{j or 'no joint'!r}"))
                    break
    for a in range(m.nu):
        an = acts[a][0]
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            out.append(("actuators", f"{an} has no joint transmission"))
        if [float(x) for x in m.actuator_gear[a]] != [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
            out.append(("actuator gear", f"{an} gear {[float(x) for x in m.actuator_gear[a]]}; it must be "
                                         "[1 0 0 0 0 0]"))
        if int(m.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE):
            out.append(("actuator dynamics", f"{an} has activation dynamics"))
        if int(m.actuator_gaintype[a]) != int(mujoco.mjtGain.mjGAIN_FIXED) or not _all_zero(m.actuator_gainprm[a, 1:]):
            out.append(("actuator gain", f"{an}: the gain is not a fixed k alone"))
        if int(m.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_AFFINE) or not _all_zero(m.actuator_biasprm[a, 1:]):
            out.append(("actuator bias", f"{an}: the bias is not an affine constant b alone (a <position> actuator, or "
                                         "a one-number biasprm inside a class that sets more terms, fails here)"))
        extra = []
        if float(m.actuator_damping[a]) != 0.0 or not _all_zero(m.actuator_dampingpoly[a]):
            extra.append("damping")
        if float(m.actuator_armature[a]) != 0.0:
            extra.append("armature")
        if float(m.actuator_delay[a]) != 0.0:
            extra.append("delay")
        if extra:
            out.append(("actuator terms", f"{an} carries actuator {', '.join(extra)}; only k and b are read"))
    # tool sites
    for side in base.SIDES:
        site, body = TOOL_SITES[side]
        sid = mujoco.mj_name2id(m, O.mjOBJ_SITE, site)
        if sid < 0:
            out.append(("tool sites", f"site {site} is missing"))
        elif _nm(m, O.mjOBJ_BODY, int(m.site_bodyid[sid])) != body:
            out.append(("tool sites", f"{site} is on {_nm(m, O.mjOBJ_BODY, int(m.site_bodyid[sid]))!r}; it must be "
                                      f"on {body}"))
    # finite numbers
    for fld in ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "jnt_pos", "jnt_axis",
                "jnt_range", "dof_armature", "dof_damping", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm",
                "site_pos", "site_quat", "eq_data"):
        if not _all_finite(getattr(m, fld)):
            out.append(("finite numbers", f"{fld} holds a non-finite number"))
    return out


# ------------------------------------------------------------------------------------------------- 5. calibration
def calibration_failures(cal):
    """Returns (failures, {arm joint: (sign, zero_offset_rad)} or None)."""
    if not isinstance(cal, dict) or not isinstance(cal.get("joints"), dict):
        return [("calibration", "calibration.json has no 'joints' object")], None
    zmax = float(caps()["zero_offset_abs_max_rad"])
    J = cal["joints"]
    out = {}
    for name in base.ARM_JOINTS:
        e = J.get(name)
        if not isinstance(e, dict):
            return [("calibration", f"{name} is missing")], None
        s, z = e.get("sign"), e.get("zero_offset_rad")
        if isinstance(s, bool) or not isinstance(s, (int, float)) or s not in (1, -1):
            return [("calibration", f"{name}: sign {s!r}; it must be +1 or -1")], None
        if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z):
            return [("calibration", f"{name}: zero_offset_rad {z!r} is not a finite number")], None
        if abs(z) > zmax:
            return [("calibration", f"{name}: zero_offset_rad {z!r} is outside +-pi")], None
        out[name] = (int(s), float(z))
    return [], out


# -------------------------------------------------------------------------------------------------------- 6. caps
def cap_failures(m):
    O = mujoco.mjtObj
    c = caps()
    out = []
    if m.nbody - 1 > int(c["max_bodies"]):
        out.append(("counts", f"{m.nbody - 1} bodies; at most {int(c['max_bodies'])}"))
    if m.ngeom > int(c["max_geoms"]):
        out.append(("counts", f"{m.ngeom} geoms; at most {int(c['max_geoms'])}"))
    lo_m, hi_m = (float(x) for x in c["body_mass_kg"])
    for i in range(1, m.nbody):
        mass = float(m.body_mass[i])
        if not lo_m <= mass <= hi_m:
            out.append(("masses", f"{_nm(m, O.mjOBJ_BODY, i)} has {mass!r} kg; the cap is {lo_m} .. {hi_m} kg"))
        inertia = sorted(float(x) for x in m.body_inertia[i])
        if not (inertia[0] > 0.0 and inertia[0] + inertia[1] >= inertia[2] * (1.0 - 1e-12)):
            out.append(("masses", f"{_nm(m, O.mjOBJ_BODY, i)} has a non-physical inertia {inertia}"))
    R = float(c["range_abs_max_rad"])
    dof_caps = (("armature", "armature_kg_m2", m.dof_armature), ("damping", "damping_Nm_s_rad", m.dof_damping),
                ("frictionloss", "frictionloss_Nm", m.dof_frictionloss))
    for j in range(m.njnt):
        jn = _nm(m, O.mjOBJ_JOINT, j)
        lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
        if not m.jnt_limited[j]:
            out.append(("joint range", f"{jn} is not limited; its range must state its two stops"))
        elif not (lo < hi and abs(lo) <= R and abs(hi) <= R):
            out.append(("joint range", f"{jn} range ({lo!r}, {hi!r}) must have lower < upper, both inside +-2 pi"))
        d = int(m.jnt_dofadr[j])
        for what, key, arr in dof_caps:
            a, b = (float(x) for x in c[key])
            if not a <= float(arr[d]) <= b:
                out.append(("joint dofs", f"{jn} {what} {float(arr[d])!r} is outside {a} .. {b}"))
    klo, khi = (float(x) for x in c["k"])
    bmax = float(c["b_abs_max_Nm"])
    for a in range(m.nu):
        an = _nm(m, O.mjOBJ_ACTUATOR, a)
        k, b = float(m.actuator_gainprm[a, 0]), float(m.actuator_biasprm[a, 0])
        if not klo <= k <= khi:
            out.append(("k", f"{an} k = {k!r}; the cap is {klo} .. {khi}"))
        if not abs(b) <= bmax:
            out.append(("b", f"{an} b = {b!r}; the cap is |b| <= {bmax} N m"))
    return out


# --------------------------------------------------------------------------------------------------- the whole check
def check(xml_bytes, cal_bytes, assets_dir=None):
    """Every stage on the two files' bytes. Returns a PolicyResult (see the module text)."""
    res = PolicyResult()
    res.xml_bytes, res.cal_bytes = bytes(xml_bytes), bytes(cal_bytes)
    if not res.set("files", file_failures(res.xml_bytes, res.cal_bytes)):
        return res
    try:
        xml_text = res.xml_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        res.set("text", [("well-formed", f"robot.xml is not UTF-8 text: {e}")])
        return res
    tf, root = text_failures(xml_text)
    cal, cerr = calibration_parse(res.cal_bytes)
    if cerr:
        tf.append(("well-formed", cerr))
    if not res.set("text", tf):
        return res
    text = grader_text(root, assets_dir)
    spec, m, warns, err = compile_spec(text)
    if err is not None:
        res.set("compile", [("compile", f"does not compile: {err}")])
        return res
    if warns:
        res.set("compile", [("compile", f"{len(warns)} compiler warning(s), the first: {warns[0]}")])
        return res
    res.set("compile", [])
    res.text, res.model = text, m
    res.set("form", form_failures(m))
    cf, calmap = calibration_failures(cal)
    res.set("calibration", cf)
    res.calibration = calmap
    res.set("caps", cap_failures(m))
    return res


def check_files(xml_path, cal_path, assets_dir=None):
    with open(xml_path, "rb") as f:
        xb = f.read()
    with open(cal_path, "rb") as f:
        cb = f.read()
    return check(xb, cb, assets_dir)


__all__ = ["PolicyError", "PolicyResult", "check", "check_files", "policy", "caps", "forbidden_elements",
           "given_mesh_files", "given_assets_dir", "grader_text", "compile_spec", "text_failures", "form_failures",
           "calibration_failures", "cap_failures", "size_failures", "arm_body_tree", "hinge_bodies", "STAGES",
           "CALIBRATION_FORMAT"]
