#!/usr/bin/env python3
"""check_twin: check your flight twin the way the grader's gates see it. It scores nothing and reads no truth.

    python3 check_twin.py [SUBMISSION]                          # default /home/user/submission
    python3 check_twin.py --flight F --depth FRAME OUT.npy      # your twin's depth frame at FRAME (and OUT.png)
    python3 check_twin.py --flight F --depth OUT.npy            # the same at each frame of the spread (below)
    python3 check_twin.py --flight F --hazard FRAME             # the hazard check on your twin's own depth frame
    python3 check_twin.py --flight F --hazard                   # the same at each frame of the spread
    python3 check_twin.py --flight F --overlay FRAME OUT.png    # your twin's depth edges over the given wide frame
    python3 check_twin.py --flight F --overlay OUT.png          # the same at each frame of the spread
    python3 check_twin.py --json                                # the report as one JSON document on stdout

What it does, with the same library the grader uses (twinprobe/, next to this file) and the same numbers
(check_twin_config.json: the public numbers that twin_contract.md and the sensor files print):
  1. the size rule of the whole submission folder (files and folders, megabytes, symbolic links and special files:
     rule Z1's part that can be checked before the session ends); a submission folder that is itself a symbolic
     link or not a folder fails rule Z1, as the grader reads it;
  2. per flight, the validity gate (rule Z2): scene.xml, its mesh files and cameras.json, with every reason and
     every line of the gate's report (geoms rebuilt opaque, materials ignored, simulator warnings); before it, as the
     grader does, the nesting cap of scene.xml's elements; as the grader copies the submission, a mesh file
     inside a folder named __pycache__ does not exist for it; and after it the cap of mesh triangles counted per
     geom (each mesh asset's triangles once for every geom that names it, once if no geom names it);
  3. the ray scene, as the grader builds it: in a child process under the grader's address-space limit (rule Z7's
     memory limit), so that a twin whose ray scene or rays do not fit inside it is reported under rule Z7 instead of
     this tool growing past it; how long the build took on this session's machine, beside rule Z7's time limit for
     the build (the grader's machine is not this session's: its time differs);
  4. cameras.json under rule L: one entry for every given frame of the flight (the frames of its frames.csv) and no
     other name, each written once (rule Z2 otherwise); per entry: is a valid pose declared (rule Z4 where the
     frame is a scored moment); is the wide or the depth camera's centre inside the solid of a geom (rule Z5 where
     the frame is a scored moment); the gate's one-flight fit of ALL your declared poses against the on-board
     estimate, read at each frame's recorded stamp minus the stated offset, with the frames whose residuals are
     largest (the grader runs the same fit on the scored moments only: rules Z3 and Z6); and how long this tool's
     own cast of one frame's depth rays took, what the flight's scored moments would take at that rate, beside the
     limit of rule Z7 (the grader's machine is not this session's: its time differs); and the scoring rules in force
     (rules L, P and S and the flight's score) in twin_contract.md's own sentences, which check_twin_config.json
     carries (this tool holds no rule text of its own);
  5. --depth: the forward depth camera's ranges in your twin from your declared pose of FRAME, cast exactly as the
     grader casts them (the depth camera placed by the two-camera transform, the ray table, meshes as their convex
     hulls, every geom opaque), written as a float32 array (rows x columns of the ray table, metres, NaN where a ray
     meets nothing or the table has no ray) and as a grey PNG beside it, with the build's and the cast's times
     beside rule Z7's limits;
  6. --hazard: the hazard check of the contract on that depth frame of your twin, for every direction of the move
     family: the stop distance or "clear". The grader runs the check on every pixel where the real depth frame
     has a return and nothing is masked as moving or as another free-flyer (not only on the pixels the depth term
     scores); this tool uses every pixel of your twin's frame, so a direction may differ;
  7. --overlay: through the wide camera's model from your declared pose of FRAME, the places where your twin's depth
     jumps (or a ray passes from one geom to another, or from a hit to a miss), drawn in red over the given wide frame.
FRAME may be any given frame of the flight whose entry holds a valid pose. Without FRAME, --depth, --hazard and
--overlay run on a spread: key_frames.per_flight frames evenly spaced over the flight's frame list (the first and the
last frame among them). The spread is a set of frames for your own checks and nothing more: it is not the set of
scored moments, which this tool does not know and no given file names. Lengths are in your twin's units (the grader
reads them as metres). Exit code 0 when every gate passes, 1 when one fails, 2 for a usage error or a missing file.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import time
import zlib

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from twinprobe import gate as tp_gate  # noqa: E402
from twinprobe import hazard as tp_hazard  # noqa: E402
from twinprobe import oneflight, posechain  # noqa: E402
from twinprobe.insidesolid import Solids  # noqa: E402
from twinprobe.raycast import RayScene  # noqa: E402
from twinprobe.widemodel import WideModel  # noqa: E402

TOOL_VERSION = "check-twin/1"
CONFIG_FORMAT = "flight-twin-check-config/1"
DEFAULT_SUB = "/home/user/submission"


class UsageError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# the task's files

class Values(dict):
    """The tool's configuration values; `read` records every key looked up (the tool's own record of what it reads).
    A missing key is a usage error, never a default."""

    def __init__(self, values):
        super().__init__(values)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            raise UsageError("the tool's configuration has no entry %r" % key) from None


def load_config(path):
    """The configuration's values, and beside them (attribute rules) the sentences of twin_contract.md that state the
    scoring rules in force: [(label, [sentence, ...])] from the document's rules_in_force, or None when the document
    carries none."""
    try:
        with open(path, "rb") as fh:
            doc = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise UsageError("the tool's configuration %s cannot be read: %s" % (path, exc)) from None
    if not isinstance(doc, dict) or doc.get("format") != CONFIG_FORMAT or not isinstance(doc.get("values"), dict):
        raise UsageError("%s is not a %s document" % (path, CONFIG_FORMAT))
    vals = Values(doc["values"])
    vals.rules = None
    if "rules_in_force" in doc:
        rules = doc["rules_in_force"]
        if not (isinstance(rules, list) and rules and all(
                isinstance(r, dict) and isinstance(r.get("rule"), str) and isinstance(r.get("sentences"), list)
                and r["sentences"] and all(isinstance(x, str) for x in r["sentences"]) for r in rules)):
            raise UsageError("%s: rules_in_force is not a list of rules, each a label and its sentences" % path)
        vals.rules = [(r["rule"], list(r["sentences"])) for r in rules]
    return vals


def read_csv(path):
    """(header, rows) of a comma-separated given: lines starting with '#' are comments, the first other line is the
    header, columns are read by name."""
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError("%s cannot be read: %s" % (path, exc)) from None
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise UsageError("%s holds no header" % path)
    header = [h.strip() for h in lines[0].split(",")]
    rows = [[c.strip() for c in ln.split(",")] for ln in lines[1:]]
    return header, rows


def frame_stamps(path):
    """{frame: recorded stamp} in the order of frames.csv."""
    header, rows = read_csv(path)
    i, j = header.index("frame"), header.index("t_s")
    return {r[i]: float(r[j]) for r in rows}


def estimate_arrays(path):
    """(times, positions, quaternions x y z w) of onboard_estimate.csv, by column name."""
    header, rows = read_csv(path)
    idx = [header.index(c) for c in ("t_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")]
    a = np.array([[float(r[k]) for k in idx] for r in rows], dtype=np.float64)
    return a[:, 0].copy(), a[:, 1:4].copy(), a[:, 4:8].copy()


class Task:
    """The given files around this tool: HOME/flights/<flight>/..., HOME/sensors/depth_rays.npy. The flights are the
    configuration's flights.names_neutral, each flight's frames those of its frames.csv (their count must be the
    configuration's flights.frames_given); the tool holds no list of scored moments."""

    def __init__(self, home, cfg):
        self.home = home
        self.cfg = cfg
        if cfg["key_frames.named"] is not False:
            raise UsageError("this tool reads a configuration whose scored moments are not named (key_frames.named "
                             "false)")
        self.flights = list(cfg["flights.names_neutral"])
        self.per_flight = int(cfg["key_frames.per_flight"])
        self._rays = None
        self._stamps = {}
        self._est = {}
        self.files_read = []

    def frames(self, flight):
        """The flight's given frames, in the order of its frames.csv."""
        names = list(self.stamps(flight))
        want = int(self.cfg["flights.frames_given"][flight])
        if len(names) != want:
            raise UsageError("%s: frames.csv lists %d frames, the configuration states %d" % (flight, len(names), want))
        return names

    def spread(self, flight):
        """key_frames.per_flight frames evenly spaced over the flight's frame list (first and last included): a spread
        for the agent's own checks, computed from the frame list and that count only."""
        names = self.frames(flight)
        k = min(self.per_flight, len(names))
        if k <= 1:
            return names[:k]
        return [names[int(round(i * (len(names) - 1) / float(k - 1)))] for i in range(k)]

    def path(self, *parts):
        return os.path.join(self.home, *parts)

    def rays(self):
        if self._rays is None:
            p = self.path("sensors", "depth_rays.npy")
            self.files_read.append(os.path.relpath(p, self.home))
            try:
                self._rays = np.load(p, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise UsageError("%s cannot be read: %s" % (p, exc)) from None
        return self._rays

    def stamps(self, flight):
        if flight not in self._stamps:
            p = self.path("flights", flight, "frames.csv")
            self.files_read.append(os.path.relpath(p, self.home))
            self._stamps[flight] = frame_stamps(p)
        return self._stamps[flight]

    def estimate(self, flight):
        if flight not in self._est:
            p = self.path("flights", flight, "onboard_estimate.csv")
            self.files_read.append(os.path.relpath(p, self.home))
            self._est[flight] = estimate_arrays(p)
        return self._est[flight]


# ---------------------------------------------------------------------------------------------------------------
# the gates

def _r(x, n=4):
    if x is None:
        return None
    x = float(x)
    return round(x, n) if math.isfinite(x) else None


# the frames whose residuals the report names (a length of a list printed, not a rule)
LARGEST = 10


def frame_rules(task, flight, res, inside_of):
    """The per-entry rules on every entry of the flight's cameras.json whose gate passed, in the grader's order per
    frame (Z4, then Z6, then Z5), with the gate's one-flight fit run on EVERY declared valid entry (the grader runs
    the same fit on the scored moments only); inside_of: {frame: (inside, which)} of the wide and the depth camera
    centres, from the ray scene's child. Nothing here knows which frames are scored."""
    cfg = task.cfg
    given = task.frames(flight)
    stamps = task.stamps(flight)
    t, p, q = task.estimate(flight)
    names = [k for k in given if k in res.poses]
    est, inside_span = [], {}
    for k in names:
        T, ins = posechain.estimate_wide_pose(t, p, q, stamps[k], cfg)
        est.append(T)
        inside_span[k] = ins
    fit = oneflight.fit_one_flight([res.poses[k] for k in names], est, oneflight.params_from_config(cfg))
    rows = []
    pos = {k: i for i, k in enumerate(names)}
    for k in given:
        row = {"frame": k, "rule": None, "e_m": None, "a_deg": None, "inlier": None, "inside": None}
        if k not in res.poses:
            row["rule"] = "Z4: " + res.frame_errors.get(k, "no valid pose")
            rows.append(row)
            continue
        i = pos[k]
        row["e_m"], row["a_deg"], row["inlier"] = _r(fit["e_m"][i]), _r(fit["a_deg"][i], 3), bool(fit["inlier"][i])
        row["estimate_inside_its_span"] = bool(inside_span[k])
        inside, which = inside_of[k]
        row["inside"] = [int(which[0]) if inside[0] else None, int(which[1]) if inside[1] else None]
        if not row["inlier"]:
            row["rule"] = "Z6: an outlier of the one-flight fit of all entries"
        elif inside.any():
            cams = [n for n, b in zip(("wide", "depth"), inside) if b]
            row["rule"] = "Z5: the %s camera centre is inside geom %s" % (
                " and the ".join(cams), ", ".join(str(int(w)) for w in which[inside]))
        rows.append(row)
    summary = oneflight.residual_summary(fit)
    fitted = [r for r in rows if r["e_m"] is not None]
    by_e = sorted(fitted, key=lambda r: (-r["e_m"], r["frame"]))[:LARGEST]
    by_a = sorted(fitted, key=lambda r: (-r["a_deg"], r["frame"]))[:LARGEST]
    return {"fit": summary, "fit_entries": len(names), "frames": rows, "given_frames": len(given),
            "largest_position_residuals": [[r["frame"], r["e_m"]] for r in by_e],
            "largest_angle_residuals": [[r["frame"], r["a_deg"]] for r in by_a],
            "entries_without_valid_pose": sum(1 for r in rows if r["rule"] and r["rule"].startswith("Z4")),
            "entries_outliers": sum(1 for r in rows if r["rule"] and r["rule"].startswith("Z6")),
            "entries_inside_a_geom": sum(1 for r in rows if r["rule"] and r["rule"].startswith("Z5")),
            "scored_moments": task.per_flight, "min_inliers": int(cfg["gate.one_flight.min_inliers"])}


def rules_in_force(task, flight):
    """The scoring rules in force, as twin_contract.md states them: one line per rule of the configuration's
    rules_in_force, its label and its sentences as they are (written there from the same source as the contract's).
    This tool holds no rule text of its own and reads no rule's setting: a configuration that carries no sentences
    gets one line that says so."""
    rules = getattr(task.cfg, "rules", None)
    if rules is None:
        return ["rule L, rule P, rule S: this configuration carries no sentence of the scoring rules; twin_contract.md "
                "states them"]
    return ["%s: %s" % (label, " ".join(sentences)) for label, sentences in rules]


# the nesting test: the same text as the grader's function of this name (the documents' closing step compares them)
def nesting_problem(flight_dir, cap, max_bytes, max_elements):
    """Rule Z2's nesting cap, read before the validity gate: the deepest level of the elements of the flight's
    scene.xml (the root element is level 1, each element one level below its parent), read by a streaming XML reader
    that builds no tree and does not recurse. Returns the reason when an element lies deeper than cap levels, else
    None. A scene.xml that is missing, not a regular file, larger than max_bytes, holding a NUL byte or a DOCTYPE or
    ENTITY declaration, holding more than max_elements elements or not well formed is left to the validity gate,
    which names its reason (the levels read before a syntax error or before the element cap still count)."""
    import os
    import stat
    import xml.parsers.expat
    path = os.path.join(flight_dir, "scene.xml")
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        with open(path, "rb") as fh:
            raw = fh.read(int(max_bytes) + 1)
    except OSError:
        return None
    up = raw.upper()
    if len(raw) > int(max_bytes) or b"\x00" in raw or b"<!DOCTYPE" in up or b"<!ENTITY" in up:
        return None
    level = [0, 0, 0]

    def start(name, attrs):
        level[0] += 1
        level[1] = max(level[1], level[0])
        level[2] += 1

    def end(name):
        level[0] -= 1

    reader = xml.parsers.expat.ParserCreate()
    reader.StartElementHandler = start
    reader.EndElementHandler = end
    step = 1 << 16
    try:
        for i in range(0, len(raw), step):
            reader.Parse(raw[i:i + step], False)
            if level[1] > int(cap) or level[2] > int(max_elements):
                break
        else:
            reader.Parse(b"", True)
    except xml.parsers.expat.ExpatError:
        pass
    if level[1] > int(cap):
        return ("scene.xml nests its elements more than %d levels deep (cap %d levels, the root element being "
                "level 1)" % (int(cap), int(cap)))
    return None


GATE_TOO_DEEP = ("the validity gate cannot read scene.xml: its elements nest too deeply for the gate's reader "
                 "(RecursionError)")


# the count of mesh triangles per geom: the same text as the grader's function of this name (the documents' closing
# step compares them)
def mesh_faces_problem(flight_dir, spec, cap, max_bytes):
    """Rule Z2's cap of mesh triangles counted per geom (caps.mesh_faces_per_flight), read after the validity gate has
    passed, on the gate's compiled copy (spec: its mesh files named relative to the flight's folder), and before
    anything is built from the compiled scene. Each mesh asset's triangles, counted from its file by the probe
    library's own counters (as the gate counts them: an OBJ polygon of n corners counts n - 2), count once for every
    geom that names the asset and once if no geom does. A geom names the asset whose name the simulator's reader gave
    the geom's mesh: its own mesh attribute, else the one of the default class the simulator applies to it (its class
    attribute, else the childclass of the nearest enclosing body that has one, else the main default; a nested class
    inherits from its parent), whatever the geom's type (the simulator fits a primitive that names a mesh to that
    mesh). The geoms inside <default> are not geoms of the scene. Returns the reason when the count is over cap, else
    None."""
    import os
    assets, counted = [], {}
    for mesh in spec.meshes:
        rel = mesh.file
        if rel not in counted:
            info = None
            path, why = tp_gate._regular_file(flight_dir, rel)
            if why is None:
                try:
                    with open(path, "rb") as fh:
                        data = fh.read(int(max_bytes) + 1)
                except OSError:
                    data = None
                if data is not None and len(data) <= int(max_bytes):
                    count = tp_gate._stl_info if os.path.splitext(rel)[1].lower() == ".stl" else tp_gate._obj_info
                    info, why = count(data)
            if info is None:
                return "mesh file %s cannot be read again to count its triangles per geom" % rel
            counted[rel] = int(info["faces"])
        assets.append((mesh.name, counted[rel]))
    named = {}
    for geom in spec.geoms:
        named[geom.meshname] = named.get(geom.meshname, 0) + 1
    total = sum(n * max(1, named.get(name, 0)) for name, n in assets)
    if total > int(cap):
        return ("the geoms name mesh assets of %d triangles in all, each asset counted once for every geom that names "
                "it and once if no geom does (cap %d)" % (total, int(cap)))
    return None


def _failed_gate(why):
    res = tp_gate.GateResult()
    res.fail(why)
    res.lines.append("validity gate (%s): FAIL" % tp_gate.GATE_VERSION)
    res.lines.append("  reason: %s" % why)
    return res


def pycache_problems(fdir):
    """The mesh files of a scene that passed the gate which the grader never sees: it copies the submission for
    grading without the folders named __pycache__, so a mesh file inside one does not exist for it (rule Z2). Returns
    [(the gate's reason as the grader gives it, the mesh path)]."""
    import xml.etree.ElementTree as ET
    try:
        with open(os.path.join(fdir, "scene.xml"), "rb") as fh:
            tree = ET.fromstring(fh.read())
    except (OSError, ET.ParseError):
        return []
    comp = tree.find("compiler")
    meshdir = comp.attrib.get("meshdir", "") if comp is not None else ""
    out = []
    for el in tree.iter("mesh"):
        f = el.attrib.get("file")
        if f is None:
            continue
        rel = tp_gate._join_rel(meshdir, f)
        if "__pycache__" in rel.split("/")[:-1]:
            why = "<mesh file=%r> (%s) does not exist" % (f, rel)
            if all(why != w for w, _ in out):
                out.append((why, rel))
    return out


# rule L's entry list of cameras.json: the same text as the grader's function of this name (the documents' closing
# step compares them)
def every_frame_cameras(flight_dir, given_frames, max_bytes, cfg):
    """Rule L (key_frames.named false), read after the validity gate's own reading of cameras.json: the file declares
    the wide camera's pose for every given frame of the flight (given_frames: the frame names of the flight's
    frames.csv, in its order) and for no other name, each frame once and each of the file's own keys once. Returns
    (reasons, entries): reasons, the rule Z2 reasons (empty when the entry list is the given frames); entries,
    {frame: why} for the given frames whose entry holds no valid pose by the per-entry rules (posechain.parse_cameras,
    the gate's own reading, run on every entry: an entry that is not an object, a position that is not three finite
    numbers, a quaternion that is not four finite numbers or whose norm lies outside gate.quaternion_norm_range). A
    cameras.json that is missing, not a regular file, larger than max_bytes, not JSON, not an object of the format or
    without its frames object is left to the validity gate, which names its reason: ([], {})."""
    import json

    class Obj(dict):
        twice = ()

    def pairs(items):
        obj, seen, twice = Obj(), set(), []
        for k, v in items:
            if k in seen and k not in twice:
                twice.append(k)
            seen.add(k)
            obj[k] = v
        obj.twice = tuple(twice)
        return obj
    path, why = tp_gate._regular_file(flight_dir, "cameras.json")
    if why is not None:
        return [], {}
    try:
        with open(path, "rb") as fh:
            raw = fh.read(int(max_bytes) + 1)
    except OSError:
        return [], {}
    if len(raw) > int(max_bytes) or posechain.parse_cameras(raw, [], cfg)[2] is not None:
        return [], {}
    obj = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    frames = obj["frames"]
    given = list(given_frames)
    known = set(given)
    reasons = []
    if obj.twice:
        reasons.append("cameras.json writes its key%s %s more than once (rule L)" % (
            "s" if len(obj.twice) > 1 else "", ", ".join(repr(k)[:40] for k in obj.twice[:5])))
    if frames.twice:
        reasons.append("cameras.json names %d frame%s more than once (rule L: one entry per given frame; first: %s)" % (
            len(frames.twice), "s" if len(frames.twice) > 1 else "", ", ".join(repr(k)[:40] for k in frames.twice[:5])))
    missing = [k for k in given if k not in frames]
    if missing:
        reasons.append("cameras.json declares no entry for %d given frame%s of the flight (rule L: one entry per given "
                       "frame of frames.csv, %d in all; first: %s)" % (
                           len(missing), "s" if len(missing) > 1 else "", len(given), ", ".join(missing[:5])))
    unknown = [k for k in frames if k not in known]
    if unknown:
        reasons.append("cameras.json holds %d entr%s for names that are no given frame of the flight (rule L: no other "
                       "entry; first: %s)" % (len(unknown), "ies" if len(unknown) > 1 else "y",
                                              ", ".join(repr(k)[:40] for k in unknown[:5])))
    _, errors, _ = posechain.parse_cameras(raw, given, cfg)
    return reasons, {k: errors[k] for k in given if k in frames and k in errors}


def _neutral_lines(lines, n_given):
    """The gate's report lines for a check of every entry: the library names the frames it was asked about 'key
    frames'; here they are the flight's given frames, and a frame's rule Z4 applies where it is a scored moment."""
    out = []
    for ln in lines:
        m = re.match(r"^  key frames with a valid pose: (\d+) of (\d+)$", ln)
        if m:
            out.append("  entries with a valid pose: %s of the %d given frames" % (m.group(1), n_given))
            continue
        if ln.startswith("  ") and ln.endswith(": rule Z4, this frame scores 0"):
            out.append(ln[:-len(": rule Z4, this frame scores 0")] + ": rule Z4 where the frame is a scored moment")
            continue
        out.append(ln)
    return out


def gate_scene(fdir, given, cfg):
    """(gate result, None). In the grader's order: the nesting cap; the validity gate (a RecursionError of its reading
    is rule Z2), here asked about every given frame of the flight; rule L's entry list of cameras.json
    (every_frame_cameras); a mesh file inside a folder named __pycache__ fails the gate with the grader's
    missing-file reason (the grader's copy of the submission leaves such folders out); the cap of mesh triangles
    counted per geom. The ray scene and the solids are built after it, in a child process (scene_work), as the grader
    builds them."""
    rules = tp_gate.rules_from_config(cfg)
    why = nesting_problem(fdir, cfg["caps.nesting_levels_per_flight"], rules["bytes"], rules["xml_elements"])
    if why is not None:
        return _failed_gate(why), None
    try:
        res = tp_gate.check_flight(fdir, given, cfg)
    except RecursionError:
        return _failed_gate(GATE_TOO_DEEP), None
    res.lines = _neutral_lines(res.lines, len(given))
    why_l, _ = every_frame_cameras(fdir, given, rules["bytes"], cfg)
    if why_l:
        if res.ok:
            res.lines[0] = "validity gate (%s): FAIL" % tp_gate.GATE_VERSION
        for w in why_l:
            res.fail(w)
            res.lines.append("  reason: %s" % w)
    if not res.ok:
        return res, None
    hidden = pycache_problems(fdir)
    if hidden:
        res.lines.append("rule Z2 as the grader copies the submission (without the folders named __pycache__): "
                         "FAIL")
        for why, rel in hidden:
            res.fail(why)
            res.lines.append("  reason: %s" % why)
            res.lines.append("    (%s lies inside a folder named __pycache__, which the grader's copy of the "
                             "submission leaves out)" % rel)
        return res, None
    why = mesh_faces_problem(fdir, res.spec, cfg["caps.mesh_faces_per_flight"], rules["mesh_bytes"])
    if why is not None:
        return _failed_gate(why), None
    return res, None


# the ray scene and its rays in a child process: the same text as the grader's function of this name (the documents'
# closing step compares them)
def scene_in_child(build, model, memory_mb, build_s, work, work_s, deadline=None):
    """Rule Z7's limits on the ray scene: build(model) -> (scene, solids) (the probe library's ray scene and solids of
    a compiled twin) and then work(scene, solids) -> a result pickle can carry (work None: build only), both in ONE
    child process (os.fork) that shares nothing with this one afterwards. The child sets its address-space limit to
    memory_mb MB and arms its own wall-clock alarm with the default action of SIGALRM, which the kernel applies even
    inside the simulator's C code: build_s s for the build, then work_s s for the work (math.inf: no alarm), each cut
    to the absolute time deadline when one is given and nearer (then own_limit is False). This process only waits and
    reads; it sends no signal. An exhausted address space is recognised in the child by what the build or the work
    raises: a MemoryError (Python's, and the simulator's binding raises it for a failed allocation of its own), or a
    RuntimeError (the simulator's binding raises one, "Caught an unknown exception!", when its compiler meets an
    allocation that fails under the limit, the hull library's included); a build that names a resource in a
    ValueError has already made it a MemoryError. Returns a dict:
      stage        "build" or "work": the step the child was in when it stopped; "done" when it finished
      failed       None; "time" (the step's alarm ended the child); "memory" (see above); "error" (any other
                   exception: its type name and text in error); "died" (the child ended by another signal or
                   without its report: error says how)
      own_limit    True when the step's alarm was the step's own limit (build_s or work_s), not the deadline
      error        (type name, text) or None
      build_s, work_s   the child's own seconds for each step (None where not reached)
      lines        the scene's report lines once built
      result       work's result or None
      peak_rss_mb  the child's peak resident memory (MB)
    """
    import math
    import os
    import pickle
    import resource
    import signal
    import struct
    import time
    import warnings

    def cut(limit):
        c, own = float(limit), True
        if deadline is not None and float(deadline) - time.time() < c:
            c, own = float(deadline) - time.time(), False
        return c, own

    rfd, wfd = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)     # a fork while idle BLAS threads exist
        pid = os.fork()
    if pid == 0:                                   # the child: it reports through the pipe and leaves by os._exit
        code = 0
        try:
            os.close(rfd)
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            lim = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))       # a child that dies leaves no core file

            def send(msg):
                data = pickle.dumps(msg, protocol=4)
                view = memoryview(struct.pack("<Q", len(data)) + data)
                while len(view):
                    view = view[os.write(wfd, view):]

            def step(name, limit, fn):
                c, own = cut(limit)
                send({"stage": name, "own_limit": own})
                if c <= 0.0:
                    send({"failed": "time", "error": None})
                    return None, False
                if math.isfinite(c):
                    signal.setitimer(signal.ITIMER_REAL, c)
                t0 = time.time()
                fail = None
                try:
                    out = fn()
                except MemoryError:
                    fail = ("memory", ("MemoryError", ""))
                except RuntimeError as exc:
                    fail = ("memory", ("RuntimeError", str(exc)[:300]))
                except Exception as exc:  # noqa: BLE001 - reported by its type and text
                    fail = ("error", (type(exc).__name__, str(exc)[:2000]))
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if fail is not None:
                    send({"failed": fail[0], "error": fail[1], name + "_s": time.time() - t0})
                    return None, False
                return (out, time.time() - t0), True

            got, ok = step("build", build_s, lambda: build(model))
            if ok:
                (scene, solids), sec = got
                got = None
                send({"built": True, "build_s": sec, "lines": list(scene.report_lines())})
                if work is not None:
                    got, ok = step("work", work_s, lambda: work(scene, solids))
                    if ok:
                        send({"result": got[0], "work_s": got[1]})
                    else:
                        scene = solids = None
                    got = None
                if ok:
                    send({"stage": "done"})
        except BaseException:  # noqa: BLE001 - the parent reads the missing report as a death of the child
            code = 1
        finally:
            os._exit(code)
    os.close(wfd)
    chunks = []
    while True:
        b = os.read(rfd, 1 << 20)
        if not b:
            break
        chunks.append(b)
    os.close(rfd)
    _, status, usage = os.wait4(pid, 0)
    data = b"".join(chunks)
    out = {"stage": "build", "failed": None, "own_limit": True, "error": None, "build_s": None, "work_s": None,
           "lines": [], "result": None, "peak_rss_mb": usage.ru_maxrss / 1024.0}
    i = 0
    while i + 8 <= len(data):
        n = struct.unpack("<Q", data[i:i + 8])[0]
        if i + 8 + n > len(data):
            break
        msg = pickle.loads(data[i + 8:i + 8 + n])
        i += 8 + n
        if "stage" in msg:
            out["stage"] = msg["stage"]
            out["own_limit"] = msg.get("own_limit", out["own_limit"])
        if msg.get("built"):
            out["build_s"], out["lines"] = msg["build_s"], msg["lines"]
        if "failed" in msg:
            out["failed"], out["error"] = msg["failed"], msg["error"]
        if "result" in msg:
            out["result"], out["work_s"] = msg["result"], msg["work_s"]
        for k in ("build_s", "work_s"):
            if k in msg and out[k] is None:
                out[k] = msg[k]
    if out["failed"] is None and out["stage"] != "done":
        if os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGALRM:
            out["failed"] = "time"
        else:
            out["failed"] = "died"
            out["error"] = ("signal %d" % os.WTERMSIG(status) if os.WIFSIGNALED(status)
                            else "exit code %d without its report" % os.WEXITSTATUS(status), "")
    return out


def build_scene(model):
    """The ray scene and the solids of a compiled twin, as the grader builds them (twinprobe's RayScene and Solids); a
    refusal of the simulator that names a resource (memory, an allocation, the hull library) is an exhausted address
    space, as the grader reads it."""
    try:
        scene = RayScene(model)
        return scene, Solids(model, scene=scene)
    except ValueError as exc:
        if any(w in str(exc).lower() for w in ("memory", "alloc", "qhull")):
            raise MemoryError(str(exc)[:200]) from None
        raise


def scene_work(task, res, work):
    """Build the twin's ray scene and run work(scene, solids) in a child process under the grader's address-space
    limit (runtime.scene_memory_limit_mb), as the grader does (scene_in_child). No time limit here: this session's
    machine is not the grader's, so the build's time is printed beside rule Z7's limit (scene_build). Returns (the
    child's outcome, the build's record, None) or (..., the reason the flight fails: rule Z2 when the simulator
    refuses to rebuild the geometry, rule Z7 when the address space is exhausted, or how the child ended)."""
    cfg = task.cfg
    mb = int(cfg["runtime.scene_memory_limit_mb"])
    out = scene_in_child(build_scene, res.model, mb, math.inf, work, math.inf)
    build = scene_build(task, out)
    if out["failed"] is None:
        return out, build, None
    if out["stage"] == "build" and out["failed"] == "error" and out["error"][0] == "ValueError":
        return out, build, ("the simulator refuses to rebuild the twin's geometry for the rays: %s"
                            % out["error"][1][:300])
    what = "build your twin's ray scene" if out["stage"] == "build" else "cast your twin's rays"
    if out["failed"] == "memory":
        return out, build, ("rule Z7: this tool could not %s inside %d MB of address space "
                            "(runtime.scene_memory_limit_mb, the grader's limit for its worker); the grader scores "
                            "such a flight 0" % (what, mb))
    err = out["error"] or ("", "")
    return out, build, ("this tool could not %s: its child process stopped (%s: %s %s); the grader's verification "
                        "would end without a score (a task error)" % (what, out["failed"], err[0], str(err[1])[:200]))


def scene_build(task, out):
    """This tool's build time of the ray scene beside rule Z7's limits (information only: no time is decided here)."""
    lim, mb = float(task.cfg["runtime.scene_time_limit_s"]), int(task.cfg["runtime.scene_memory_limit_mb"])
    sec = out["build_s"]
    if out["stage"] == "build" and out["failed"]:
        head = "this tool's build of your twin's ray scene stopped (%s)%s" % (
            out["failed"], "" if sec is None else " after %.2f s on this session's machine" % sec)
    else:
        head = "this tool built your twin's ray scene in %.2f s on this session's machine" % sec
    return {"seconds": None if sec is None else round(float(sec), 2), "rule_z7_limit_s": lim, "rule_z7_memory_mb": mb,
            "peak_rss_mb": round(float(out["peak_rss_mb"]), 1),
            "note": ("%s; rule Z7: a flight whose ray scene the grader does not build within %g s of its own wall "
                     "clock, or whose ray scene or rays do not fit inside %d MB of its worker's address space, scores "
                     "0. The grader's machine is not this session's machine, so the grader's time differs from this "
                     "figure" % (head, lim, mb))}


def cast_time(task, flight, frame, seconds, rays):
    """This tool's cast time of one frame beside rule Z7's limit (information only: no rule is decided here)."""
    n = task.per_flight
    limit = float(task.cfg["runtime.rays_time_limit_s"])
    return {"frame": frame, "rays": int(rays), "seconds": round(float(seconds), 2), "scored_moments": n,
            "all_scored_moments_s": round(float(seconds) * n, 1), "rule_z7_limit_s": limit,
            "note": ("this tool cast every ray of the table from your pose of %s in %.2f s on this session's machine; "
                     "at that rate the flight's %d scored moments would take about %.0f s; rule Z7: a flight whose "
                     "rays the grader does not cast within %g s of its own wall clock scores 0. The grader's machine "
                     "is not this session's machine, so the grader's time differs from these figures"
                     % (frame, float(seconds), n, float(seconds) * n, limit))}


def check(task, sub):
    out = {"tool": TOOL_VERSION, "submission": sub, "z1": None, "flights": {}, "ok": True}
    if os.path.islink(sub):
        # as the grader reads it: a submission folder that is a symbolic link is not read (rule Z1)
        out["z1"] = ["the submission folder %s is a symbolic link: the grader does not read it" % sub]
        out["ok"] = False
        return out
    if os.path.lexists(sub) and not os.path.isdir(sub):
        out["z1"] = ["the submission folder %s is not a folder" % sub]
        out["ok"] = False
        return out
    if not os.path.lexists(sub):
        # as the grader reads it: no folder breaks rule Z1, and each flight then has no scene.xml (rule Z2)
        out["notes"] = ["the submission folder %s does not exist: each flight below fails rule Z2" % sub]
    why = tp_gate.submission_size(sub, task.cfg)
    out["z1"] = list(why)
    if why:
        out["ok"] = False
    for f in task.flights:
        fl = {"rules_in_force": rules_in_force(task, f)}
        res, why = gate_scene(os.path.join(sub, f), task.frames(f), task.cfg)
        fl.update({"gate_ok": bool(res.ok), "reasons": list(res.reasons), "lines": list(res.lines)})
        if not fl["gate_ok"]:
            out["ok"] = False
            out["flights"][f] = fl
            continue
        kf = task.frames(f)
        k0 = next((k for k in task.spread(f) if k in res.poses), None)
        T_dw = posechain.T_depth_wide(task.cfg)

        def work(scene, solids):
            inside = {k: solids.contains(np.array([res.poses[k][:3, 3],
                                                   posechain.depth_pose(res.poses[k], T_dw)[:3, 3]]))
                      for k in kf if k in res.poses}
            cast = None
            if k0 is not None:
                t0 = time.perf_counter()
                twin_depth(task, scene, solids, res.poses[k0])
                cast = time.perf_counter() - t0
            return inside, cast
        child, fl["scene_build"], why = scene_work(task, res, work)
        if why is not None:
            refused = why.startswith("the simulator refuses")                     # rule Z2, as the grader reads it
            fl["gate_ok"] = not refused
            fl["reasons"].append(why)
            if child["failed"] == "memory":
                fl["z7"] = why
            out["ok"] = False
            out["flights"][f] = fl
            continue
        fl["lines"] += ["  " + ln for ln in child["lines"]]
        inside, cast = child["result"]
        fl.update(frame_rules(task, f, res, inside))
        if k0 is not None:
            fl["cast_time"] = cast_time(task, f, k0, cast, np.all(np.isfinite(task.rays()), axis=2).sum())
        out["flights"][f] = fl
    return out


# ---------------------------------------------------------------------------------------------------------------
# a declared pose, the twin's depth frame and the wide view

def declared_pose(task, fdir, frame):
    if frame not in task.stamps(os.path.basename(fdir)):
        raise UsageError("%s is no given frame of %s" % (frame, os.path.basename(fdir)))
    with open(os.path.join(fdir, "cameras.json"), "rb") as fh:
        raw = fh.read()
    poses, errors, file_error = posechain.parse_cameras(raw, [frame], task.cfg)
    if file_error:
        raise UsageError(file_error)
    if frame not in poses:
        raise UsageError("cameras.json declares no valid pose for %s (%s)" % (frame, errors.get(frame, "")))
    return poses[frame]


def gated(task, sub, flight):
    """(the flight's folder, its gate result), the gate run once per flight and call of the tool."""
    if flight not in task.flights:
        raise UsageError("no flight %r (the flights are %s)" % (flight, ", ".join(task.flights)))
    fdir = os.path.join(sub, flight)
    memo = task.__dict__.setdefault("_gated", {})
    if fdir not in memo:
        res, _ = gate_scene(fdir, task.frames(flight), task.cfg)
        if not res.ok:
            raise UsageError("the validity gate fails for %s: %s" % (flight, "; ".join(res.reasons[:5])))
        memo[fdir] = res
    return fdir, memo[fdir]


def in_scene(task, res, flight, work):
    """work(scene, solids) on the twin's ray scene in the child process (scene_work); (result, the build's record)."""
    child, build, why = scene_work(task, res, work)
    if why is not None:
        raise UsageError("%s: %s" % (flight, why))
    return child["result"], build


def twin_depth(task, scene, solids, T_tw):
    """(ranges (H, W) with NaN for a miss or no ray, zero reason or None) of the twin's depth frame."""
    T_td = posechain.depth_pose(T_tw, posechain.T_depth_wide(task.cfg))
    inside, which = solids.contains(np.array([T_tw[:3, 3], T_td[:3, 3]]))
    zero = None
    if inside.any():
        zero = "Z5: a camera centre is inside geom %s" % ", ".join(str(int(w)) for w in which[inside])
    rays = task.rays()
    H, W = rays.shape[:2]
    rng = np.full((H, W), np.nan)
    ok = np.all(np.isfinite(rays), axis=2)
    dirs = rays[ok] @ T_td[:3, :3].T
    dist, _, _ = scene.cast(T_td[:3, 3], dirs, float(task.cfg["gate.ray_cutoff"]))
    rng[ok] = np.where(dist >= 0.0, dist, np.nan)
    return rng, zero


def run_depth(task, sub, flight, frame, out_path):
    fdir, res = gated(task, sub, flight)
    T_tw = declared_pose(task, fdir, frame)

    def work(scene, solids):
        t0 = time.perf_counter()
        got = twin_depth(task, scene, solids, T_tw)
        return got, time.perf_counter() - t0
    ((rng, zero), seconds), build = in_scene(task, res, flight, work)
    np.save(out_path, rng.astype(np.float32), allow_pickle=False)
    png = os.path.splitext(out_path)[0] + ".png"
    lo, hi = float(task.cfg["depth.scored_range_min"]), float(task.cfg["depth.scored_range_max"])
    g = np.where(np.isfinite(rng), 255.0 - 230.0 * np.clip((rng - lo) / (hi - lo), 0.0, 1.0), 0.0)
    write_png(png, np.round(g).astype(np.uint8))
    fin = np.isfinite(rng)
    return {"flight": flight, "frame": frame, "zero": zero, "npy": out_path, "png": png,
            "rays": int(np.all(np.isfinite(task.rays()), axis=2).sum()), "hits": int(fin.sum()),
            "range_min": _r(np.nanmin(rng)) if fin.any() else None,
            "range_median": _r(np.nanmedian(rng)) if fin.any() else None,
            "range_max": _r(np.nanmax(rng)) if fin.any() else None, "scene_build": build,
            "cast_time": cast_time(task, flight, frame, seconds, np.all(np.isfinite(task.rays()), axis=2).sum())}


def run_hazard(task, sub, flight, frame):
    fdir, res = gated(task, sub, flight)
    T_tw = declared_pose(task, fdir, frame)
    (rng, zero), build = in_scene(task, res, flight, lambda scene, solids: twin_depth(task, scene, solids, T_tw))
    cfg = task.cfg
    hz = tp_hazard.HazardCheck.from_config(cfg)
    dirs = hz.family(cfg)
    rays = task.rays()
    mask = np.isfinite(rng).reshape(-1)
    k = hz.stop_indices(np.nan_to_num(rng, nan=-1.0).reshape(-1), np.nan_to_num(rays).reshape(-1, 3), mask, dirs)
    step = float(cfg["hazard.path_sample"])
    az, el = list(cfg["hazard.directions_azimuth"]), list(cfg["hazard.directions_elevation"])
    rows = []
    for i in range(len(dirs)):
        e, a = el[i // len(az)], az[i % len(az)]
        stop = None if k[i] < 0 else _r(k[i] * step, 3)
        clear = {str(L): bool(hz.is_clear(k[i], L)) for L in cfg["hazard.lengths"]}
        rows.append({"direction": i, "azimuth_deg": a, "elevation_deg": e, "stop_m": stop, "clear": clear})
    return {"flight": flight, "frame": frame, "zero": zero, "directions": rows, "scene_build": build,
            "note": "every pixel of your twin's frame is used; the grader uses only the pixels where the real depth "
                    "frame has a return and nothing is masked"}


def run_overlay(task, sub, flight, frame, out_path, stride, jump):
    fdir, res = gated(task, sub, flight)
    T_tw = declared_pose(task, fdir, frame)
    cfg = task.cfg
    wm = WideModel.from_config(cfg)
    png = task.path("flights", flight, "frames", frame + ".png")
    task.files_read.append(os.path.relpath(png, task.home))
    img = read_png_grey(png)
    H, W = img.shape
    if (W, H) != (wm.width, wm.height):
        raise UsageError("the frame is %d x %d, the camera sheet says %d x %d" % (W, H, wm.width, wm.height))
    us = np.arange(0, W, stride, dtype=float)
    vs = np.arange(0, H, stride, dtype=float)
    U, V = np.meshgrid(us, vs)
    d = wm.ray(np.stack([U.ravel(), V.ravel()], axis=1))
    ok = np.all(np.isfinite(d), axis=1)
    dirs = d[ok] @ T_tw[:3, :3].T
    (dist, src, _), build = in_scene(task, res, flight,
                                     lambda scene, solids: scene.cast(T_tw[:3, 3], dirs, float(cfg["gate.ray_cutoff"])))
    rng = np.full(len(d), np.nan)
    gid = np.full(len(d), -2, dtype=np.int64)
    rng[ok] = np.where(dist >= 0.0, dist, np.nan)
    gid[ok] = np.where(dist >= 0.0, src, -1)
    rng = rng.reshape(U.shape)
    gid = gid.reshape(U.shape)
    edge = np.zeros(U.shape, bool)
    for axis in (0, 1):
        a = np.take(rng, np.arange(rng.shape[axis] - 1), axis=axis)
        b = np.take(rng, np.arange(1, rng.shape[axis]), axis=axis)
        ga = np.take(gid, np.arange(gid.shape[axis] - 1), axis=axis)
        gb = np.take(gid, np.arange(1, gid.shape[axis]), axis=axis)
        with np.errstate(invalid="ignore"):
            e = (ga != gb) | (np.abs(a - b) > jump)
        sl = [slice(None), slice(None)]
        sl[axis] = slice(0, rng.shape[axis] - 1)
        edge[tuple(sl)] |= e
    rgb = np.repeat(img[:, :, None], 3, axis=2).astype(np.uint8)
    big = np.kron(edge.astype(np.uint8), np.ones((stride, stride), np.uint8))[:H, :W].astype(bool)
    rgb[big] = (255, 0, 0)
    write_png_rgb(out_path, rgb)
    return {"flight": flight, "frame": frame, "png": out_path, "stride_px": stride, "jump": jump,
            "rays": int(ok.sum()), "hits": int(np.isfinite(rng).sum()), "edge_cells": int(edge.sum()),
            "scene_build": build}


# ---------------------------------------------------------------------------------------------------------------
# PNG (standard library only: 8-bit grey in, 8-bit grey or RGB out)

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def read_png_grey(path):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise UsageError("%s cannot be read: %s" % (path, exc)) from None
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise UsageError("%s is not a PNG file" % path)
    pos, idat, w = 8, [], None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or ctype != 0 or interlace != 0:
                raise UsageError("%s: only 8-bit grey, non-interlaced PNG files are read here" % path)
        elif typ == b"IDAT":
            idat.append(body)
        elif typ == b"IEND":
            break
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8).reshape(h, w + 1)
    out = np.zeros((h, w), dtype=np.int32)
    prev = np.zeros(w, dtype=np.int32)
    for y in range(h):
        ft, line = int(raw[y, 0]), raw[y, 1:].astype(np.int32)
        if ft == 0:
            cur = line
        elif ft == 2:
            cur = (line + prev) & 255
        elif ft in (1, 3, 4):
            cur = np.zeros(w, dtype=np.int32)
            left = 0
            for x in range(w):
                if ft == 1:
                    pred = left
                elif ft == 3:
                    pred = (left + int(prev[x])) // 2
                else:
                    up, ul = int(prev[x]), int(prev[x - 1]) if x else 0
                    pa, pb, pc = abs(up - ul), abs(left - ul), abs(left + up - 2 * ul)
                    pred = left if (pa <= pb and pa <= pc) else (up if pb <= pc else ul)
                left = (int(line[x]) + pred) & 255
                cur[x] = left
        else:
            raise UsageError("%s: unknown PNG filter %d" % (path, ft))
        out[y] = cur
        prev = cur
    return out.astype(np.uint8)


def _chunk(typ, body):
    return struct.pack(">I", len(body)) + typ + body + struct.pack(">I", zlib.crc32(typ + body) & 0xFFFFFFFF)


def _write_png(path, arr, ctype):
    h, w = arr.shape[:2]
    rows = arr.reshape(h, -1)
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(h))
    data = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(data)


def write_png(path, grey):
    _write_png(path, np.ascontiguousarray(grey, dtype=np.uint8), 0)


def write_png_rgb(path, rgb):
    _write_png(path, np.ascontiguousarray(rgb, dtype=np.uint8), 2)


# ---------------------------------------------------------------------------------------------------------------
# the report

def print_report(rep):
    z1 = rep.get("z1") or []
    print("check_twin (%s): %s" % (TOOL_VERSION, rep["submission"]))
    print("rule Z1 (size, links, special files): %s" % ("PASS" if not z1 else "FAIL"))
    for w in z1:
        print("  " + w)
    for w in rep.get("notes") or []:
        print(w)
    for f, fl in rep["flights"].items():
        print("")
        print("%s:" % f)
        for ln in fl["lines"]:
            print("  " + ln)
        shown = "\n".join(fl["lines"])
        for w in fl["reasons"]:
            if ("reason: " + w) not in shown:
                print("  reason: " + w)
        if fl.get("scene_build"):
            print("  ray scene build time (rule Z7): " + fl["scene_build"]["note"])
        if "fit" not in fl:
            print_rules(fl.get("rules_in_force", []))
            continue
        fit = fl["fit"]
        print("  one-flight fit of all %d declared entries with a valid pose: %d agree with the on-board estimate "
              "(scale %.4f). The grader runs this fit on the %d scored moments only: fewer than %d of them agreeing "
              "is rule Z3 (the flight scores 0), a scored moment that does not agree is rule Z6 (it scores 0)" % (
                  fl["fit_entries"], fit["n_inliers"], fit["scale"], fl["scored_moments"], fl["min_inliers"]))
        print("    largest position residuals: " + ", ".join("%s %.3f m" % (k, e) for k, e in
                                                             fl["largest_position_residuals"]))
        print("    largest angle residuals: " + ", ".join("%s %.2f deg" % (k, a) for k, a in
                                                          fl["largest_angle_residuals"]))
        for what, key, rule in (("no valid pose", "entries_without_valid_pose", "Z4"),
                                ("an outlier of the fit of all entries", "entries_outliers", "Z6"),
                                ("a camera centre inside a geom", "entries_inside_a_geom", "Z5")):
            rows = [r for r in fl["frames"] if r["rule"] and r["rule"].startswith(rule)]
            print("  entries with %s: %d of %d%s" % (what, fl[key], fl["given_frames"], (
                " (rule %s where the frame is a scored moment): %s%s" % (
                    rule, ", ".join(r["frame"] for r in rows[:LARGEST]), " ..." if len(rows) > LARGEST else ""))
                if rows else ""))
        if fl.get("cast_time"):
            print("  ray cast time (rule Z7): " + fl["cast_time"]["note"])
        print_rules(fl.get("rules_in_force", []))
    print("")
    print("overall: %s" % ("every gate passes" if rep["ok"] else "a gate fails"))


def print_rules(lines):
    """The scoring rules in force as the configuration carries them: one line per rule, its label and its sentences."""
    if lines:
        print("  the scoring rules in force, in twin_contract.md's own sentences:")
    for ln in lines:
        print("    " + ln)


def spread_note(task, flight):
    sp = task.spread(flight)
    return {"frames": sp, "note": (
        "no frame named: the tool runs at a spread of %d frames evenly spaced over the %d given frames of %s (%s). "
        "The spread is for your own checks only: it is not the set of scored moments, which this tool does not know"
        % (len(sp), len(task.frames(flight)), flight, ", ".join(sp)))}


def per_frame(task, flight, args, fn):
    """args [FRAME, OUT] (one frame) or [OUT] (each frame of the spread, OUT_<frame> beside OUT)."""
    if len(args) == 2:
        return fn(args[0], args[1])
    stem, ext = os.path.splitext(args[0])
    return [fn(fr, "%s_%s%s" % (stem, fr, ext)) for fr in task.spread(flight)]


def print_mode(k, v):
    if k == "hazard":
        print("hazard check on your twin's depth frame (%s %s)%s" % (
            v["flight"], v["frame"], "; " + v["zero"] if v["zero"] else ""))
        print("  " + v["note"])
        for r in v["directions"]:
            print("  direction %2d  azimuth %6.1f  elevation %6.1f  %s" % (
                r["direction"], r["azimuth_deg"], r["elevation_deg"],
                "clear" if r["stop_m"] is None else "stop at %.2f" % r["stop_m"]))
    else:
        print("%s: %s" % (k, json.dumps(v, sort_keys=True)))
    if v.get("scene_build"):
        print("ray scene build time (rule Z7): " + v["scene_build"]["note"])
    if k == "depth":
        print("ray cast time (rule Z7): " + v["cast_time"]["note"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check a flight twin the way the grader's gates see it.")
    ap.add_argument("submission", nargs="?", default=DEFAULT_SUB)
    ap.add_argument("--flight", default=None, help="the flight for --depth, --hazard and --overlay")
    ap.add_argument("--depth", nargs="+", metavar="ARG", help="[FRAME] OUT.npy: without FRAME, at each frame of the "
                                                                "spread (OUT_<frame>.npy)")
    ap.add_argument("--hazard", nargs="?", const="", metavar="FRAME", help="without FRAME, at each frame of the spread")
    ap.add_argument("--overlay", nargs="+", metavar="ARG", help="[FRAME] OUT.png: without FRAME, at each frame of the "
                                                                  "spread (OUT_<frame>.png)")
    ap.add_argument("--stride", type=int, default=2, help="with --overlay: one ray every STRIDE pixels (default 2)")
    ap.add_argument("--jump", type=float, default=0.1,
                    help="with --overlay: a range step that counts as an edge, in your twin's units (default 0.1)")
    ap.add_argument("--home", default=os.path.dirname(HERE), help="the folder that holds flights/ and sensors/")
    ap.add_argument("--config", default=os.path.join(HERE, "check_twin_config.json"))
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of the text report")
    a = ap.parse_args(argv)
    try:
        task = Task(a.home, load_config(a.config))
        if a.stride < 1:
            raise UsageError("--stride must be at least 1")
        flight = a.flight or (task.flights[0] if len(task.flights) == 1 else None)
        extra = {}
        modes = a.depth is not None or a.hazard is not None or a.overlay is not None
        if modes:
            if flight is None:
                raise UsageError("--flight is needed")
            if flight not in task.flights:
                raise UsageError("no flight %r (the flights are %s)" % (flight, ", ".join(task.flights)))
        for name, val in (("--depth", a.depth), ("--overlay", a.overlay)):
            if val is not None and len(val) not in (1, 2):
                raise UsageError("%s takes [FRAME] OUT" % name)
        if modes and ((a.depth is not None and len(a.depth) == 1) or a.hazard == "" or
                      (a.overlay is not None and len(a.overlay) == 1)):
            extra["spread"] = spread_note(task, flight)
        if a.depth is not None:
            extra["depth"] = per_frame(task, flight, a.depth, lambda fr, out: run_depth(task, a.submission, flight,
                                                                                        fr, out))
        if a.hazard is not None:
            frames = [a.hazard] if a.hazard else task.spread(flight)
            got = [run_hazard(task, a.submission, flight, fr) for fr in frames]
            extra["hazard"] = got[0] if a.hazard else got
        if a.overlay is not None:
            extra["overlay"] = per_frame(task, flight, a.overlay, lambda fr, out: run_overlay(
                task, a.submission, flight, fr, out, a.stride, a.jump))
        rep = check(task, a.submission) if not extra else {"tool": TOOL_VERSION, "submission": a.submission,
                                                            "ok": True, "flights": {}}
        rep.update(extra)
    except UsageError as exc:
        print("check_twin: %s" % exc, file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    elif extra:
        if "spread" in extra:
            print(extra["spread"]["note"])
        for k, v in extra.items():
            if k == "spread":
                continue
            for one in (v if isinstance(v, list) else [v]):
                print_mode(k, one)
    else:
        print_report(rep)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
