"""quad_task: the robot specification and the grader's constants, shared by verify.py, quad_convert.py and quad_grade.py.

The vocabulary is fixed: legs FL FR HL HR; the root link `body`; per leg the links <L>_shoulder, <L>_upper, <L>_lower
and the joints <L>_HAA (body -> shoulder), <L>_HFE (shoulder -> upper), <L>_KFE (upper -> lower). Everything else is
read, never assumed: which legs exist, the tags of each link, the parts' files and placements, the cores and foot pads,
the joint axes, points and limits, the foot points (the reference folder, format quad-cad-gt/1); the bought components
(given/core.json, given/foot.json and their STEP files); the converter's knobs (grader_config.json over the defaults
below, GRADER_SPEC v1; the grader's own knobs and defaults are quad_grade.py's). Lengths mm, angles rad in the files.

    spec = load_spec(reference_dir)            RobotSpec (raises SpecError: the task's own folder is broken)
    cfg = load_config(stage_dir or path)       DEFAULT_CONFIG deep-updated with grader_config.json
    settings = conversion_settings(cfg)        the converter's knobs (conversion + the shared consistency block)
    comp = load_component(given_dir, "core")   core.json + absolute paths + defaults (output point / axis)
    pairs = component_classes(given_dir)       [[class, step path]] for fcstd_extract.py

Pure stdlib + numpy; no FreeCAD, no geometry kernel.
"""
import copy
import json
import os
import re

import numpy as np

FORMAT = "quad-cad-gt/1"
ROOT_LINK = "body"
LEGS = ("FL", "FR", "HL", "HR")
JOINT_KINDS = ("HAA", "HFE", "KFE")
LINK_KINDS = ("shoulder", "upper", "lower")
TAG_PATTERN = r"^P(0[1-9]|[1-3][0-9]|4[0-2])$"
TAG_RE = re.compile(TAG_PATTERN)
COMPONENT_CLASSES = ("core", "foot")

# The knobs the converter and verify.py read from grader_config.json (the grader's own defaults live in quad_grade.py;
# the key names are the file's). `consistency` is shared with the grader (GRADER_SPEC 3.6); `conversion` is the
# converter's alone (GRADER_SPEC 3.5, 3.7).
DEFAULT_CONFIG = {
    "consistency": {"axis_angle_deg": 1.0, "axis_offset_mm": 0.5, "contact_mm": 0.2, "gap_search_mm": 5.0},
    "conversion": {"core_axis_deg": 5.0,          # candidate cores: +z within this of the joint axis (either sign)
                   "core_line_mm": 2.0,           # ... and on its line within this (else the nearest line, inconsistent)
                   "seed_axis_mm": 50.0,          # Kabsch seed: output centre and the point this far along +z
                   "haa_inlier_mm": 30.0,         # a leg's HAA points fit the seed within this (else an outlier leg:
                                                  # swapped or flipped legs are >= 100 mm off, honest length errors less)
                   "icp_max_mm": 15.0, "icp_max_deg": 15.0,
                   "align_samples_agent": 4000,   # per body part, fixed seed
                   "align_samples_gt": 20000,
                   "align_seed": 20261027,
                   "workers": 8,                  # parts tessellated in parallel (the verify stage's 8 CPUs)
                   "pool_min_total_mb": 2.0,      # ... when their shape files total at least this (else in-process)
                   "part_timeout_s": 300.0},      # the parts not tessellated within this (all together) are not valid
    "time_limits_s": {"extract": 900, "convert": 600, "grade": 900, "verify": 2700},
    "extract_timeout_s": 900,
}


class SpecError(RuntimeError):
    """The reference folder (the task's own ground truth) does not load: a task error, never the agent's."""


def leg_links(leg):
    return ["%s_%s" % (leg, k) for k in LINK_KINDS]


def leg_joints(leg):
    return ["%s_%s" % (leg, k) for k in JOINT_KINDS]


def leg_of(name):
    """The leg of a link or joint name ('FL_upper' -> 'FL'), None for the body."""
    head = str(name).split("_")[0]
    return head if head in LEGS else None


def _m4(a, what):
    m = np.asarray(a, float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise SpecError("%s is not a 4x4 matrix" % what)
    return m


def _v3(a, what):
    v = np.asarray(a, float).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise SpecError("%s is not a 3-vector" % what)
    return v


class RobotSpec(object):
    """The reference robot (quad-cad-gt/1). Attributes:

    root, links {name: {name, parent, joint, origin_mm (np 3)}}, link_order (depth-first: body, then leg by leg),
    joints {name: {name, parent_link, child_link, axis (unit np 3), point_mm (np 3), driven_by, limits_rad [lo, hi]}},
    joint_order (depth-first, MuJoCo's qpos order), legs (present, in LEGS order), tags (sorted), link_tags
    {link: sorted tags}, tag_link {tag: link}, parts {tag: record + T_link np 4x4 + path}, cores {id: {id, link, T_link}},
    core_of_joint {joint: core id}, feet {id: {id, link, T_link}}, foot_of_leg {leg: foot id}, foot_points {leg: np 3}
    (lower-leg link frame).
    """

    def __init__(self, ref_dir, raw):
        self.dir = ref_dir
        self.raw = raw
        if raw.get("format") != FORMAT:
            raise SpecError("assembly.json format is %r (expected %s)" % (raw.get("format"), FORMAT))
        self.root = raw.get("root_link", ROOT_LINK)
        if self.root != ROOT_LINK:
            raise SpecError("root link %r (expected %s)" % (self.root, ROOT_LINK))
        self.links = {}
        for l in raw.get("links", []):
            name = l["name"]
            if name in self.links:
                raise SpecError("link %s listed twice" % name)
            self.links[name] = {"name": name, "parent": l.get("parent"), "joint": l.get("joint"),
                                "origin_mm": _v3(l.get("origin_mm", [0, 0, 0]), "link %s origin_mm" % name)}
        if self.root not in self.links:
            raise SpecError("no root link %s" % self.root)
        self.legs = [leg for leg in LEGS if any(n in self.links for n in leg_links(leg))]
        expected = {self.root: (None, None)}
        for leg in self.legs:
            sh, up, lo = leg_links(leg)
            haa, hfe, kfe = leg_joints(leg)
            expected.update({sh: (self.root, haa), up: (sh, hfe), lo: (up, kfe)})
        if set(self.links) != set(expected):
            raise SpecError("links %s differ from the vocabulary %s" % (sorted(self.links), sorted(expected)))
        for name, (parent, joint) in expected.items():
            l = self.links[name]
            if l["parent"] != parent or l["joint"] != joint:
                raise SpecError("link %s: parent %s / joint %s (expected %s / %s)" % (name, l["parent"], l["joint"], parent, joint))
        self.link_order = [self.root] + [n for leg in self.legs for n in leg_links(leg)]
        self.joint_order = [j for leg in self.legs for j in leg_joints(leg)]
        self.joints = {}
        for j in raw.get("joints", []):
            name = j["name"]
            if name not in self.joint_order:
                raise SpecError("joint %s is not in the vocabulary" % name)
            child = self.links.get(j.get("child_link"), {})
            if child.get("joint") != name or child.get("parent") != j.get("parent_link"):
                raise SpecError("joint %s: parent %s / child %s disagree with the links" % (name, j.get("parent_link"), j.get("child_link")))
            axis = _v3(j["axis"], "joint %s axis" % name)
            n = float(np.linalg.norm(axis))
            if abs(n - 1.0) > 1e-3:
                raise SpecError("joint %s axis is not a unit vector (%.6f)" % (name, n))
            lim = [float(v) for v in j.get("limits_rad", [-np.pi, np.pi])]
            if len(lim) != 2 or not lim[0] < lim[1]:
                raise SpecError("joint %s limits %s" % (name, lim))
            self.joints[name] = {"name": name, "parent_link": j["parent_link"], "child_link": j["child_link"],
                                 "axis": axis / n, "point_mm": _v3(j.get("point_mm", child["origin_mm"]), "joint %s point" % name),
                                 "driven_by": j.get("driven_by"), "limits_rad": lim}
        if set(self.joints) != set(self.joint_order):
            raise SpecError("joints %s differ from the vocabulary %s" % (sorted(self.joints), self.joint_order))
        self.parts, self.tag_link, link_tags = {}, {}, {n: [] for n in self.links}
        for p in raw.get("parts", []):
            tag = p["tag"]
            if not TAG_RE.match(tag):
                raise SpecError("part tag %r does not match %s" % (tag, TAG_PATTERN))
            if tag in self.parts:
                raise SpecError("part tag %s listed twice" % tag)
            if p.get("link") not in self.links:
                raise SpecError("part %s: unknown link %r" % (tag, p.get("link")))
            path = os.path.normpath(os.path.join(ref_dir, p["file"]))
            rec = dict(p)
            rec["T_link"] = _m4(p.get("T_link", np.eye(4)), "part %s T_link" % tag)
            rec["path"] = path
            self.parts[tag] = rec
            self.tag_link[tag] = p["link"]
            link_tags[p["link"]].append(tag)
        self.tags = sorted(self.parts)
        self.link_tags = {n: sorted(t) for n, t in link_tags.items()}
        empty = [n for n, t in self.link_tags.items() if not t]
        if empty:
            raise SpecError("links without parts: %s (a link is identified by its tags)" % empty)
        self.cores = {}
        for c in raw.get("cores", []):
            if c.get("link") not in self.links:
                raise SpecError("core %s: unknown link %r" % (c.get("id"), c.get("link")))
            self.cores[c["id"]] = {"id": c["id"], "link": c["link"], "T_link": _m4(c["T_link"], "core %s T_link" % c["id"])}
        self.core_of_joint = {}
        for name, j in self.joints.items():
            cid = j["driven_by"]
            if cid not in self.cores:
                raise SpecError("joint %s is driven by %r, which is not among the cores" % (name, cid))
            if self.cores[cid]["link"] != j["parent_link"]:
                raise SpecError("joint %s: its core %s sits in %s, not in the parent link %s" % (name, cid, self.cores[cid]["link"], j["parent_link"]))
            self.core_of_joint[name] = cid
        self.feet, self.foot_of_leg = {}, {}
        for f in raw.get("feet", []):
            if f.get("link") not in self.links:
                raise SpecError("foot %s: unknown link %r" % (f.get("id"), f.get("link")))
            self.feet[f["id"]] = {"id": f["id"], "link": f["link"], "T_link": _m4(f["T_link"], "foot %s T_link" % f["id"])}
            leg = leg_of(f["link"])
            if leg is None or f["link"] != leg_links(leg)[2]:
                raise SpecError("foot %s sits in %s, not in a lower leg" % (f["id"], f["link"]))
            if leg in self.foot_of_leg:
                raise SpecError("leg %s has two foot pads" % leg)
            self.foot_of_leg[leg] = f["id"]
        self.foot_points = {}
        for leg, pt in (raw.get("foot_points") or {}).items():
            self.foot_points[leg] = _v3(pt, "foot point %s" % leg)
        for leg in self.legs:
            if leg not in self.foot_of_leg:
                raise SpecError("leg %s has no foot pad" % leg)
            if leg not in self.foot_points:
                self.foot_points[leg] = self.feet[self.foot_of_leg[leg]]["T_link"][:3, 3].copy()

    # -- helpers
    def children(self, link):
        return [n for n in self.link_order if self.links[n]["parent"] == link]

    def link_origin(self, link):
        return self.links[link]["origin_mm"].copy()

    def joint_of_link(self, link):
        return self.links[link]["joint"]

    def legs_of_links(self):
        return {n: leg_of(n) for n in self.links}

    def part_T_robot(self, tag):
        """The part file's frame -> the robot frame at q = 0 (link frames are world-aligned translations)."""
        T = np.eye(4)
        T[:3, 3] = self.links[self.tag_link[tag]]["origin_mm"]
        return T @ self.parts[tag]["T_link"]

    def core_T_robot(self, core_id):
        T = np.eye(4)
        T[:3, 3] = self.links[self.cores[core_id]["link"]]["origin_mm"]
        return T @ self.cores[core_id]["T_link"]

    def foot_T_robot(self, foot_id):
        T = np.eye(4)
        T[:3, 3] = self.links[self.feet[foot_id]["link"]]["origin_mm"]
        return T @ self.feet[foot_id]["T_link"]


def load_spec(reference_dir):
    path = os.path.join(reference_dir, "assembly.json")
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SpecError("reference %s does not load: %s" % (path, exc))
    try:
        return RobotSpec(reference_dir, raw)
    except SpecError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecError("reference %s is malformed: %r" % (path, exc))


def _deep_update(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path=None):
    """DEFAULT_CONFIG deep-updated with grader_config.json (`path` = the file or the folder holding it; None or a missing
    file = the defaults). Every other key of the file is kept as it is (the grader's)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    if os.path.isdir(path):
        path = os.path.join(path, "grader_config.json")
    if not os.path.isfile(path):
        return cfg
    with open(path) as fh:
        extra = json.load(fh)
    return _deep_update(cfg, extra)


def conversion_settings(cfg=None):
    """The converter's flat settings: `conversion` plus the shared `consistency` thresholds (as consistent_axis_deg,
    consistent_offset_mm, contact_mm, contact_search_mm)."""
    cfg = cfg or {}
    out = copy.deepcopy(DEFAULT_CONFIG["conversion"])
    _deep_update(out, cfg.get("conversion", {}))
    cons = copy.deepcopy(DEFAULT_CONFIG["consistency"])
    _deep_update(cons, cfg.get("consistency", {}))
    out.update(consistent_axis_deg=float(cons["axis_angle_deg"]), consistent_offset_mm=float(cons["axis_offset_mm"]),
               contact_mm=float(cons["contact_mm"]), contact_search_mm=float(cons["gap_search_mm"]))
    return out


def load_component(given_dir, cls):
    """The component file given/<cls>.json (format quad-component/1) with absolute `step_path` (and `keepout_path`)
    and the output defaults (point [0, 0, 0], axis [0, 0, 1]: the core frame's own); a missing JSON gives the STEP
    alone (given/<cls>.step)."""
    meta = {"format": "quad-component/1", "class": cls, "step": cls + ".step"}
    p = os.path.join(given_dir, cls + ".json")
    if os.path.isfile(p):
        with open(p) as fh:
            meta.update(json.load(fh))
    meta["step_path"] = os.path.join(given_dir, meta.get("step") or cls + ".step")
    if meta.get("keepout_step"):
        meta["keepout_path"] = os.path.join(given_dir, meta["keepout_step"])
    out = dict(meta.get("output") or {})
    out["point"] = [float(v) for v in out.get("point", [0.0, 0.0, 0.0])]
    ax = np.asarray(out.get("axis", [0.0, 0.0, 1.0]), float)
    out["axis"] = (ax / np.linalg.norm(ax)).tolist()
    meta["output"] = out
    return meta


def component_classes(given_dir, classes=COMPONENT_CLASSES):
    """[[class, STEP path]] of the bought components present in `given_dir` (the extractor's CLASSES argument)."""
    out = []
    for cls in classes:
        meta = load_component(given_dir, cls)
        if os.path.isfile(meta["step_path"]):
            out.append([cls, meta["step_path"]])
    return out
