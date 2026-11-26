"""One submission through the grader: per flight the validity gate, the one-flight fit, the frame rules, the depth
term and the hazard term; then the aggregations and the task's score.

    ctx = Context(cfg, givens, truths)
    res = grade_submission(root, ctx, guard)       # JSON-ready; res["twin_score"] before the integrity gate G1
    check_reference(res, ctx)                      # the reference must score exactly 1 in every term (else TruthError)

The scored moments (the key frames) are the truth's own index (twin/truth.py). With key_frames.named false (rule L of
the second version) no given file names them: cameras.json declares a pose for every given frame of the flight, the
gate checks that entry list (every_frame_cameras) and the per-entry rules on every entry, and from there on only the
entries of the scored moments are read, exactly as the first version read its declared key frames. With
key_frames.named true (the first version) the configuration names them and cameras.json is read at them only.
Per flight:
  gate   rule Z2: first the nesting cap (nesting_problem: scene.xml's elements no deeper than
         caps.nesting_levels_per_flight levels, read without recursion), then twinprobe.gate.check_flight (scene.xml,
         its mesh files, cameras.json), then the cap of mesh triangles counted per geom (mesh_faces_problem, on the
         gate's compiled copy, before the ray scene is built: each mesh asset's triangles once for every geom that
         names it, once if no geom does); with key_frames.named false, rule L's entry list of cameras.json
         (every_frame_cameras, after the gate's own reading of the file); a failure, or a RecursionError of the
         gate's reading: the flight is 0
  fit    rules Z3 / Z6: the one-flight fit of the declared poses against the on-board estimate read at each frame's
         stamp minus the configured offset (it reads the poses only: it runs before the scene, and its outcome is
         applied after the scene is built, as the order of the rules has it); fewer inliers than the configured
         minimum: the flight is 0
  scene  the ray scene and the solids of the compiled copy (every geom opaque, meshes as their convex hulls), built
         in a child process (scene_in_child) under the worker's address-space limit; the simulator refusing to
         rebuild a geometry is the submission's fault (rule Z2): the flight is 0; rule Z7: when the scene is not
         built inside the stage's own limit (runtime.scene_time_limit_s, the runtime block's scene cap) or inside the
         worker's address space (runtime.scene_memory_limit_mb, the runtime block's address-space limit) the flight
         is 0
  rays   in the same child, per key frame, in order: Z4 (no valid pose), Z6 (an outlier of the fit), Z5 (the wide or
         the depth camera's centre inside a geom's solid); otherwise the depth frame's ranges, cast in row chunks;
         rule Z7: when the casting does not finish inside the stage's own limit (runtime.rays_time_limit_s, the
         runtime block's ray cap) or inside that address space the flight is 0. A time-out at the verification's
         deadline, the child ending by any other signal, and any other error of the two stages stay void
  hazard the hazard check on the twin's frame at the truth's hazard pixels, every direction of the family
  score  the depth term and the hazard term from the frames' ranges and stop indices; every registered aggregation
         of each (score.hazard_aggregation, score.depth_aggregation) is computed, the ones in force are H and D
A frame zeroed by Z4, Z5 or Z6 scores 0 on every scored pixel and every move. Formulas: score.py.
"""
import math

import numpy as np

from . import score
from .config import ConfigError
from .givens import GivenError
from .guard import GraderVoid, SubmissionFault
from .truth import CLASSES, TruthError
from .twinprobe import gate as tp_gate
from .twinprobe import hazard as tp_hazard
from .twinprobe import oneflight, posechain
from .twinprobe.insidesolid import Solids
from .twinprobe.raycast import RayScene

GRADE_FORMAT = "flight-twin-grade/1"
# every rule that zeroes the submission, a flight, a frame or a scored item, for the documents' generator (which
# asserts that the contract lists exactly these); G1 is applied by verify.py, the rest here and in score.py
ZERO_RULES = (
    ("G1", "submission", "the integrity gate: an entry written after the session ended; more entries or bytes than "
                         "caps.files and caps.megabytes; a symbolic link or a special file in the submission; a "
                         "submission root that is a link, not a folder, outside the home or replaced after the session"),
    ("Z2", "flight", "the validity gate (twinprobe.gate): scene.xml missing, not valid, using an element or attribute "
                     "off the allow list, not compiling or over a cap (the mesh triangles counted per geom: each "
                     "asset once for every geom that names it, once if no geom does); its elements nested deeper than "
                     "caps.nesting_levels_per_flight levels (read before the gate), or too deep for the gate's reader; "
                     "a mesh file missing or malformed; cameras.json missing or malformed; with key_frames.named "
                     "false, cameras.json lacking an entry for a given frame of the flight, holding an entry for a "
                     "name that is no given frame of the flight, or naming a frame or one of its keys twice (rule L); "
                     "or the simulator refusing to rebuild the compiled geometry for the rays"),
    ("Z3", "flight", "fewer key frames agree with the on-board estimate than gate.one_flight.min_inliers"),
    ("Z4", "frame", "no valid pose declared (none, a number not finite, a quaternion norm outside "
                    "gate.quaternion_norm_range)"),
    ("Z5", "frame", "the wide or the depth camera's centre inside the solid of a geom"),
    ("Z6", "frame", "the frame's declared pose is an outlier of the one-flight fit"),
    ("Z7", "flight", "the grader does not build the flight's ray scene inside runtime.scene_time_limit_s of its wall "
                     "clock, or does not cast the flight's rays inside runtime.rays_time_limit_s of its wall clock, or "
                     "either does not fit inside runtime.scene_memory_limit_mb of the worker's address space"),
    ("S1", "pixel or move", "a ray of the twin that hits nothing; an error at the band's zero end or beyond; a blocked "
                            "move the twin's frame finds clear, or whose stop differs by hazard.stop_band_zero or more"),
    ("S2", "flight", "the structure credit enters the depth term only multiplied by the content credit"),
    ("S3", "flight", "J = max(0, aC + aBC - 1): no hazard credit unless clear and content-blocked moves are told apart"),
)
TERMS = ("D", "Dc", "Ds", "D_weighted", "D_product", "H", "H_moves", "H_poses", "H_clean_poses", "J", "aC", "aBC",
         "aBS", "sum", "product", "score")
# the counts of rule P per flight (score.pose_term) and of rule L (entries of cameras.json without a valid pose)
POSE_COUNTS = ("key_poses", "key_poses_kept", "key_poses_kept_clean")


class Context:
    """What every grade needs, built once from the configuration, the givens and the truth."""

    def __init__(self, cfg, givens, truths):
        self.cfg = cfg
        self.givens = givens
        self.truths = truths
        self.flights = list(cfg["flights.names_neutral"])
        if len(self.flights) != cfg["flights.count_first_version"] or len(set(self.flights)) != len(self.flights):
            raise ConfigError("flights.names_neutral does not hold flights.count_first_version distinct names")
        if cfg["gate.opaque_forced"] is not True or not cfg["gate.mesh_cast"].startswith("convex hull"):
            raise ConfigError("the configuration's gate rules are not the ones the ray scene applies")
        self.T_dw = posechain.T_depth_wide(cfg)
        self.cutoff = float(cfg["gate.ray_cutoff"])
        self.hz = tp_hazard.HazardCheck.from_config(cfg)
        self.dirs = self.hz.family(cfg)
        self.band = score.stop_band_samples(cfg)
        self.chunk_rows = int(cfg.runtime["ray_chunk_rows"])
        self.rays_limit = float(cfg["runtime.rays_time_limit_s"])
        if float(cfg.runtime["stage_caps_s"]["rays"]) != self.rays_limit:
            raise ConfigError("the runtime block's ray cap is not runtime.rays_time_limit_s")
        self.scene_limit = float(cfg["runtime.scene_time_limit_s"])
        if float(cfg.runtime["stage_caps_s"]["scene"]) != self.scene_limit:
            raise ConfigError("the runtime block's scene cap is not runtime.scene_time_limit_s")
        self.memory_limit = int(cfg["runtime.scene_memory_limit_mb"])
        if int(cfg.runtime["worker_address_space_mb"]) != self.memory_limit:
            raise ConfigError("the runtime block's address-space limit is not runtime.scene_memory_limit_mb")
        self.fit_params = oneflight.params_from_config(cfg)
        self.k_lengths = {float(L): self.hz.length_index(L) for L in cfg["hazard.lengths"]}
        self.k_max = self.hz.length_index(cfg["hazard.max_length"])
        self.named = cfg["key_frames.named"]
        self.hazard_aggregation = cfg["score.hazard_aggregation"]
        self.depth_aggregation = cfg["score.depth_aggregation"]
        self.given = {}                     # flight -> its given frames (the names of frames.csv, in its order)
        counts = cfg["flights.frames_given"]
        for f in self.flights:
            if f not in truths:
                raise TruthError("no truth for %s" % f)
            names = list(givens.flights[f].stamps)
            if len(names) != int(counts[f]):
                raise GivenError("%s: frames.csv lists %d frames, flights.frames_given states %d"
                                 % (f, len(names), int(counts[f])))
            if any(k not in givens.flights[f].stamps for k in truths[f].key_frames):
                raise TruthError("%s: a key frame of the truth is no given frame of the flight" % f)
            self.given[f] = names


# ---------------------------------------------------------------------------------------------------------------
# stages (each called through guard.run)

# the nesting test: the same text as the check tool's function of this name (the documents' closing step compares them)
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


# the count of mesh triangles per geom: the same text as the check tool's function of this name (the documents'
# closing step compares them)
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


# rule L's entry list of cameras.json: the same text as the check tool's function of this name (the documents' closing
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


def _gate(fdir, key_frames, cfg, given=None):
    """Rule Z2: the nesting cap (before anything parses scene.xml into a tree), then the validity gate, then (given:
    the flight's given frames, when the scored moments are not named) rule L's entry list of cameras.json, then the
    cap of mesh triangles counted per geom (on the gate's compiled copy, before the ray scene is built). A
    RecursionError raised by the gate's reading is the submission's fault (Z2), never void. The gate reads the poses
    of key_frames only, as before; res.entries_without_pose: rule L's {frame: why} over every entry (given only)."""
    rules = tp_gate.rules_from_config(cfg)
    why = nesting_problem(fdir, cfg["caps.nesting_levels_per_flight"], rules["bytes"], rules["xml_elements"])
    if why is not None:
        res = tp_gate.GateResult()
        res.fail(why)
        res.lines.append("validity gate (%s): FAIL" % tp_gate.GATE_VERSION)
        res.lines.append("  reason: %s" % why)
        res.entries_without_pose = {}
        return res
    try:
        res = tp_gate.check_flight(fdir, key_frames, cfg)
    except RecursionError:
        raise SubmissionFault(GATE_TOO_DEEP) from None
    res.entries_without_pose = {}
    if given is not None:
        why_l, res.entries_without_pose = every_frame_cameras(fdir, given, rules["bytes"], cfg)
        if why_l:
            if res.ok:
                res.lines[0] = "validity gate (%s): FAIL" % tp_gate.GATE_VERSION
            for w in why_l:
                res.fail(w)
                res.lines.append("  reason: %s" % w)
    if res.ok:
        why = mesh_faces_problem(fdir, res.spec, cfg["caps.mesh_faces_per_flight"], rules["mesh_bytes"])
        if why is not None:
            entries = res.entries_without_pose
            res = tp_gate.GateResult()
            res.entries_without_pose = entries
            res.fail(why)
            res.lines.append("validity gate (%s): FAIL" % tp_gate.GATE_VERSION)
            res.lines.append("  reason: %s" % why)
    return res


def _scene(model):
    """The ray scene and the solids of a compiled twin. The simulator refusing to rebuild a geometry it has just
    compiled is the submission's fault (SubmissionFault), unless its message points at a resource (memory, an
    allocation, the hull library, which reports an exhausted address space as its own error): that voids."""
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    try:
        sc = RayScene(model, d)
        return sc, Solids(model, d, scene=sc)
    except ValueError as exc:
        msg = str(exc)
        if any(w in msg.lower() for w in ("memory", "alloc", "qhull")):
            raise MemoryError("the simulator could not rebuild the geometry (%s)" % msg[:200]) from None
        raise SubmissionFault("the simulator refuses to rebuild the twin's geometry for the rays: %s" % msg[:300]) from None


def _fit(poses, ft, gv, cfg, params):
    names = [k for k in ft.key_frames if k in poses]
    est = [posechain.estimate_wide_pose(gv.times, gv.positions, gv.quats_xyzw, gv.stamp(k), cfg)[0] for k in names]
    res = oneflight.fit_one_flight([poses[k] for k in names], est, params)
    inl = {k: bool(res["inlier"][i]) for i, k in enumerate(names)}
    return res, inl


def _rays(scene, solids, poses, inlier, ft, rays, ctx):
    """Frame rules and ranges per key frame: [(zero reason or None, ranges (H, W) or None)]."""
    out = []
    H = rays.shape[0]
    for i, k in enumerate(ft.key_frames):
        if k not in poses:
            out.append(("Z4", None))
            continue
        if not inlier.get(k, False):
            out.append(("Z6", None))
            continue
        T_tw = poses[k]
        T_td = posechain.depth_pose(T_tw, ctx.T_dw)
        inside, _ = solids.contains(np.array([T_tw[:3, 3], T_td[:3, 3]]))
        if inside.any():
            out.append(("Z5", None))
            continue
        need = ft.hazard[i]
        rng = np.full(need.shape, np.nan)
        R = T_td[:3, :3]
        for a in range(0, H, ctx.chunk_rows):
            m = need[a:a + ctx.chunk_rows]
            if not m.any():
                continue
            dirs = rays[a:a + ctx.chunk_rows][m] @ R.T
            dist, _, _ = scene.cast(T_td[:3, 3], dirs, ctx.cutoff)
            block = rng[a:a + ctx.chunk_rows]
            block[m] = np.where(dist >= 0.0, dist, np.nan)
        out.append((None, rng))
    return out


def _hazard(frames, ft, rays, ctx):
    """First colliding sample per direction of the family for each frame with ranges (None for a zeroed frame)."""
    R = rays.reshape(-1, 3)
    out = []
    for i, (zero, rng) in enumerate(frames):
        if zero is not None:
            out.append(None)
            continue
        out.append(ctx.hz.stop_indices(rng.reshape(-1), R, ft.hazard[i].reshape(-1), ctx.dirs))
    return out


def _terms(frames, stops, frame_errors, ft, ctx, counts, zero_frames):
    """The depth term and the hazard term of a flight from its frames' ranges and stop indices."""
    cfg = ctx.cfg
    per_frame = []
    detail = {"frames": [], "moves": {c: [] for c in CLASSES}}
    for i, k in enumerate(ft.key_frames):
        zero, rng = frames[i]
        c_m, s_m = ft.content[i], ft.structure[i]
        if zero is None:
            cc = score.depth_credit(rng[c_m], ft.depth[i][c_m], cfg)
            cs = score.depth_credit(rng[s_m], ft.depth[i][s_m], cfg)
            counts["scored_frames"] += 1
        else:
            cc = np.zeros(int(c_m.sum()))
            cs = np.zeros(int(s_m.sum()))
            counts[zero.lower()] += 1
            why = frame_errors.get(k, "") if zero == "Z4" else ""
            zero_frames[k] = zero + (": " + why if why else "")
        per_frame.append((cc, cs))
        detail["frames"].append({"frame": k, "zero": zero, "content_pixels": int(len(cc)),
                                 "structure_pixels": int(len(cs)),
                                 "content_mean": float(cc.mean()) if len(cc) else None,
                                 "structure_mean": float(cs.mean()) if len(cs) else None})
    dt = score.depth_term(per_frame, cfg)
    index = {k: i for i, k in enumerate(ft.key_frames)}
    credits = {c: [] for c in CLASSES}
    by_pose = {k: [] for k in ft.key_frames}
    for m in ft.moves:
        st = stops[index[m["key_frame"]]]
        if st is None:
            cr = 0.0
        else:
            kt = int(st[m["direction"]])
            if m["class"] == "clear":
                cr = 1.0 if (kt < 0 or kt > ctx.k_lengths[float(m["length"])]) else 0.0
            else:
                cr = score.stop_credit(kt, int(round(m["stop"] / cfg["hazard.path_sample"])), ctx.band)
        credits[m["class"]].append(cr)
        detail["moves"][m["class"]].append(cr)
        by_pose[m["key_frame"]].append(cr)
    ht = score.hazard_term(credits, cfg)
    pt = score.pose_term([(k, by_pose[k]) for k in ft.key_frames if by_pose[k]])
    hz = score.hazard_aggregations(ht["H"], pt["H_poses"], pt["H_clean_poses"])
    H = hz[ctx.hazard_aggregation]
    ag = score.aggregate(H, dt["D"], cfg)
    terms = {k: dt[k] for k in ("D", "Dc", "Ds", "D_weighted", "D_product")}
    terms.update({k: ht[k] for k in ("J", "aC", "aBC", "aBS")})
    terms.update({"H": H, "H_moves": ht["H"], "H_poses": pt["H_poses"], "H_clean_poses": pt["H_clean_poses"]})
    terms.update(ag)
    for k in POSE_COUNTS:
        counts[k] = pt[k]
    # the maintainers' record of rules P and S (the grade dump only): the pose rows, every aggregation and the flight's score
    # under every pair of them (the flight aggregation in force)
    detail["poses"] = pt["poses"]
    detail["poses_without_moves"] = [k for k in ft.key_frames if not by_pose[k]]
    dforms = score.depth_aggregations(dt["Dc"], dt["Ds"], cfg)
    detail["aggregations"] = {
        "in_force": {"hazard": ctx.hazard_aggregation, "depth": ctx.depth_aggregation,
                     "flight": cfg["score.flight_aggregation"]},
        "hazard": hz, "depth": dforms,
        "score": {"%s|%s" % (h, d): score.aggregate(hz[h], dforms[d], cfg)["score"] for h in sorted(hz)
                  for d in sorted(dforms)}}
    return terms, detail


# the ray scene and its rays in a child process: the same text as the check tool's function of this name (the
# documents' closing step compares them)
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


def _limit_reason(stage, out, ctx):
    """Rule Z7's reason for a step of scene_in_child ("scene": the build, "rays": the casting) that stopped at the
    step's own limit or by an exhausted address space. Any other stop voids the verification (GraderVoid): the
    verification's deadline, the child ending by another signal or without its report, any other error."""
    what = "building of this flight's ray scene" if stage == "scene" else "casting of this flight's rays"
    if out["failed"] == "time" and out["own_limit"]:
        if stage == "rays":
            return ("Z7: the grader's casting of this flight's rays did not finish inside its limit of %g s of the "
                    "grader's wall clock (runtime.rays_time_limit_s): the flight scores 0" % ctx.rays_limit)
        return ("Z7: the grader's building of this flight's ray scene did not finish inside its limit of %g s of the "
                "grader's wall clock (runtime.scene_time_limit_s): the flight scores 0" % ctx.scene_limit)
    if out["failed"] == "memory":
        return ("Z7: the grader's %s ran out of the worker's address space of %d MB (runtime.scene_memory_limit_mb): "
                "the flight scores 0" % (what, ctx.memory_limit))
    if out["failed"] == "time":
        raise GraderVoid("stage %s ran out of the verify's time: the verification is void (no outcome may come from "
                         "the clock)" % stage)
    if out["failed"] == "died":
        raise GraderVoid("stage %s: the process that builds the ray scene and casts its rays ended with %s: the "
                         "verification is void" % (stage, out["error"][0]))
    raise GraderVoid("stage %s: internal error %s: %s" % (stage, out["error"][0], str(out["error"][1])[:300]))


# ---------------------------------------------------------------------------------------------------------------
# one flight, one submission

def zero_terms():
    return {k: 0.0 for k in TERMS}


def grade_flight(fdir, name, ctx, guard, entry_rule=None):
    """One flight. entry_rule: None, the configuration's (rule L's every-frame entry list when key_frames.named is
    false); "key_frames": the first version's (cameras.json read at the scored moments only, other entries ignored),
    which the self-test applies to the frozen reference twin, a first-version file of the scored moments."""
    cfg = ctx.cfg
    ft = ctx.truths[name]
    gv = ctx.givens.flights[name]
    rays = ctx.givens.rays
    K = len(ft.key_frames)
    if entry_rule not in (None, "key_frames"):
        raise ValueError("entry_rule %r" % (entry_rule,))
    every_frame = entry_rule is None and not ctx.named
    res = {"flight": name, "G0": 0.0, "z3": False, "z7": False, "reasons": [], "lines": [], "terms": zero_terms(),
           "counts": {"key_frames": K, "valid_poses": 0, "inliers": 0, "z4": 0, "z5": 0, "z6": 0, "scored_frames": 0,
                      "key_poses": 0, "key_poses_kept": 0, "key_poses_kept_clean": 0,
                      "entries_without_valid_pose": 0},
           "zero_frames": {}, "fit": None, "detail": None,
           "cameras": {"entry_rule": "every given frame (rule L)" if every_frame else "the key frames",
                       "given_frames": len(ctx.given[name]) if every_frame else None,
                       "entries_without_valid_pose_not_scored": {}}}
    try:
        g = guard.run("gate", _gate, fdir, ft.key_frames, cfg, ctx.given[name] if every_frame else None)
    except SubmissionFault as exc:
        res["reasons"].append("G0: %s" % exc)
        return res
    res["lines"] = list(g.lines)
    if every_frame:
        wrong = g.entries_without_pose
        res["counts"]["entries_without_valid_pose"] = len(wrong)
        res["cameras"]["entries_without_valid_pose_not_scored"] = {k: v for k, v in wrong.items()
                                                                  if k not in ft.key_frames}
    if not g.ok:
        res["reasons"] = ["G0: %s" % r for r in g.reasons[:20]]
        return res
    # the fit reads the declared poses only; its outcome is applied after the scene is built, as before
    fit, inlier = guard.run("fit", _fit, g.poses, ft, gv, cfg, ctx.fit_params)

    def cast(scene, solids):
        return _rays(scene, solids, g.poses, inlier, ft, rays, ctx)
    out = scene_in_child(_scene, g.model, ctx.memory_limit, guard.limit("scene"), None if fit["z3"] else cast,
                         guard.limit("rays"), guard.deadline if guard.enabled else None)
    guard.stats[name] = {k: out[k] for k in ("stage", "failed", "own_limit", "build_s", "work_s", "peak_rss_mb")}
    guard.stats[name]["error"] = out["error"][0] if out["error"] else None
    if out["stage"] == "build" and out["failed"] == "error" and out["error"][0] == "SubmissionFault":
        res["reasons"].append("G0: the twin's geometry cannot be rebuilt for the rays (%s)" % out["error"][1])
        return res
    scene_z7 = _limit_reason("scene", out, ctx) if out["stage"] == "build" and out["failed"] else None
    res["lines"] += ["  " + ln for ln in out["lines"]]
    res["G0"] = 1.0
    res["counts"]["valid_poses"] = len(g.poses)
    res["fit"] = oneflight.residual_summary(fit)
    res["counts"]["inliers"] = int(fit["n_inliers"])
    if scene_z7 is not None:
        res["z7"] = True
        res["reasons"].append(scene_z7)
        return res
    if fit["z3"]:
        res["z3"] = True
        res["reasons"].append("Z3: %d key frames agree with the on-board estimate, fewer than %d: the declared "
                              "poses are not one place" % (int(fit["n_inliers"]), ctx.fit_params["min_inliers"]))
        return res
    if out["failed"]:
        res["z7"] = True
        res["reasons"].append(_limit_reason("rays", out, ctx))
        return res
    frames = out["result"]
    stops = guard.run("hazard", _hazard, frames, ft, rays, ctx)
    terms, detail = guard.run("score", _terms, frames, stops, g.frame_errors, ft, ctx, res["counts"],
                              res["zero_frames"])
    res["terms"].update(terms)
    res["detail"] = detail
    for z in ("z4", "z5", "z6"):
        if res["counts"][z]:
            res["reasons"].append("%s: %d key frame(s) score 0" % (z.upper(), res["counts"][z]))
    if every_frame and res["counts"]["entries_without_valid_pose"]:
        res["reasons"].append("L: %d of the %d entries of cameras.json hold no valid pose (rule Z4 where one is a scored "
                              "moment, no effect elsewhere)" % (res["counts"]["entries_without_valid_pose"],
                                                                 len(ctx.given[name])))
    return res


def grade_submission(root, ctx, guard, entry_rule=None):
    """Every flight of a submission folder (root/<flight>/); twin_score before the integrity gate. entry_rule: see
    grade_flight (None for every submission; the self-test's frozen reference twin is read with "key_frames")."""
    import os
    out = {"format": GRADE_FORMAT, "flights": {}, "aggregation": ctx.cfg["score.flight_aggregation"],
           "hazard_aggregation": ctx.hazard_aggregation, "depth_aggregation": ctx.depth_aggregation,
           "key_frames_named": bool(ctx.named)}
    for name in ctx.flights:
        out["flights"][name] = grade_flight(os.path.join(root, name), name, ctx, guard, entry_rule=entry_rule)
    n = float(len(ctx.flights))
    for key in ("score", "sum", "product"):
        out["twin_score" if key == "score" else "twin_score_" + key] = float(
            sum(f["G0"] * f["terms"][key] for f in out["flights"].values()) / n)
    return out


def check_reference(res, ctx):
    """The reference must score exactly 1 in every term of every flight (every hazard and depth aggregation among
    them), under both flight aggregations, with every key frame an inlier, no frame zeroed, every key pose with a
    scored move kept under both readings of rule P and no rule Z7; anything else is a TruthError."""
    bad = []
    for name, f in res["flights"].items():
        K = f["counts"]["key_frames"]
        if f["G0"] != 1.0 or f["z3"] or f["z7"]:
            bad.append("%s: G0 %s, Z3 %s, Z7 %s (%s)" % (name, f["G0"], f["z3"], f["z7"], "; ".join(f["reasons"][:3])))
            continue
        if f["counts"]["inliers"] != K or f["counts"]["scored_frames"] != K:
            bad.append("%s: %d inliers and %d scored frames of %d (%s)" % (
                name, f["counts"]["inliers"], f["counts"]["scored_frames"], K, f["zero_frames"]))
        c = f["counts"]
        if not (c["key_poses"] > 0 and c["key_poses_kept"] == c["key_poses"] == c["key_poses_kept_clean"]):
            bad.append("%s: key poses kept %d (clean %d) of %d" % (name, c["key_poses_kept"],
                                                                   c["key_poses_kept_clean"], c["key_poses"]))
        off = {k: v for k, v in f["terms"].items() if v != 1.0}
        if off:
            bad.append("%s: terms not exactly 1: %s" % (name, off))
    for key in ("twin_score", "twin_score_sum", "twin_score_product"):
        if res[key] != 1.0:
            bad.append("%s = %r" % (key, res[key]))
    if bad:
        raise TruthError("the reference twin does not score exactly 1: " + " | ".join(bad))
    return True


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
