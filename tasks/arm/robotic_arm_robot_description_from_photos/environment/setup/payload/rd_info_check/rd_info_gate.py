"""rd_info.gate: the contract gate `model_loads` of the robot-description stage (payload-safe; public rules only).

The contract of robot_contract.md checked on a submission folder (no assembly.json exists: the arm is measured from its
photos). Reads no ground truth, so `check_submission.py` may run it on any submission folder.

    ok, details, models = gate_model_loads(sub_dir)      # models = {"robot": MjModel, "scene": MjModel, "calibration": dict}

Order: files present -> file hygiene (regular files only, byte cap) -> the MJCF whitelist on scene.xml (with its include)
and robot.xml BEFORE anything is compiled -> the file-path rule (every `file=` and every `meshdir` / `texturedir` /
`assetdir` a relative path without a `..` component, so that whatever base MuJoCo resolves it against -- the model's
directory, meshdir, texturedir or assetdir -- stays inside the submission; the shared whitelist resolves textures through
meshdir, which MuJoCo does not) -> explicit header rules (angle unit, include) -> zero-warning compile of
scene.xml and robot.xml -> contract structure (six hinges in chain order, the seven bodies with the contract parents
(extra jointless bodies allowed), no `ref`, sites, `home`, six position actuators) -> calibration.json -> caps.
Every failure reason is listed in details["violations"].
"""

import json
import math
import os
import re
import stat as _stat
import tempfile
import xml.etree.ElementTree as ET

try:
    from . import contract as C
except ImportError:          # vendored flat (the grader's copy and the payload): `import rd_info_contract as C`
    import rd_info_contract as C  # type: ignore

GATE_VERSION = "rd-info-gate/1.1.0"


def submission_files(sub_dir):
    """(regular files with sizes, irregular entries) below sub_dir; symlinks and special files are irregular."""
    regular, irregular = {}, []
    for root, dirs, files in os.walk(sub_dir):
        for name in list(dirs):
            p = os.path.join(root, name)
            if os.path.islink(p):
                irregular.append(os.path.relpath(p, sub_dir) + " (symlinked directory)")
                dirs.remove(name)
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.lstat(p)
            except OSError:
                irregular.append(os.path.relpath(p, sub_dir) + " (unreadable)")
                continue
            if _stat.S_ISLNK(st.st_mode):
                irregular.append(os.path.relpath(p, sub_dir) + " (symlink)")
            elif not _stat.S_ISREG(st.st_mode):
                irregular.append(os.path.relpath(p, sub_dir) + " (not a regular file)")
            else:
                regular[os.path.relpath(p, sub_dir)] = int(st.st_size)
    return regular, irregular


def compile_model(xml_path):
    """(model | None, warnings, error): MuJoCo's warning callback AND fd-level stderr capture."""
    import mujoco
    warnings, error = [], None
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    model = None
    try:
        try:
            mujoco.set_mju_user_warning(lambda msg: warnings.append(str(msg)))
        except Exception:  # noqa: BLE001
            pass
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:600]
        finally:
            try:
                mujoco.set_mju_user_warning(None)
            except Exception:  # noqa: BLE001
                pass
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", "replace")
        tmp.close()
    if "WARNING" in captured.upper():
        warnings.append(captured.strip()[:400])
    return model, warnings, error


def _names(m, objtype, n):
    import mujoco
    return [mujoco.mj_id2name(m, objtype, i) for i in range(n)]


def body_groups(model):
    """{body id: contract body name | "world"}: every compiled body belongs to its nearest contract ancestor-or-self
    (a jointless sub-body of a link is part of that link; bodies above `base` belong to "world")."""
    import mujoco
    names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    out = {}
    for i in range(model.nbody):
        k = i
        g = "world"
        while k > 0:
            if names[k] in C.BODIES:
                g = names[k]
                break
            k = int(model.body_parentid[k])
        out[i] = g
    return out


def collision_geoms_by_group(model, groups=None):
    """{contract body: [geom ids]} of the geoms that can collide (contype or conaffinity non-zero)."""
    groups = groups or body_groups(model)
    out = {}
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        out.setdefault(groups[int(model.geom_bodyid[g])], []).append(g)
    return out


def load_calibration(path):
    """(calibration dict | None, violations)."""
    v = []
    try:
        with open(path) as f:
            cal = json.load(f)
    except (OSError, ValueError) as exc:
        return None, ["calibration.json does not parse: {}".format(str(exc)[:120])]
    js = cal.get("joints") if isinstance(cal, dict) else None
    if not isinstance(js, dict):
        return None, ['calibration.json needs {"joints": {"<joint>": {"sign": 1 | -1, "zero_offset_rad": float}}}']
    out = {}
    for j in C.JOINTS:
        e = js.get(j)
        if not isinstance(e, dict):
            v.append("calibration.json: joint {} missing".format(j))
            continue
        s, z = e.get("sign"), e.get("zero_offset_rad")
        if isinstance(s, bool) or s not in (1, -1, 1.0, -1.0):
            v.append("calibration.json: {} sign {!r} must be +1 or -1".format(j, s))
            continue
        if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(float(z)):
            v.append("calibration.json: {} zero_offset_rad {!r} must be a finite number".format(j, z))
            continue
        out[j] = {"sign": int(s), "zero_offset_rad": float(z)}
    extra = [k for k in js if k not in C.JOINTS]
    if extra:
        v.append("calibration.json: unknown joints {}".format(extra[:6]))
    return ({"joints": out} if not v else None), v


PATH_ATTRS = ("meshdir", "texturedir", "assetdir")


def _bad_path(v):
    """Why a file / directory attribute may leave the submission (None when it cannot)."""
    v = str(v)
    if not v.strip():
        return None
    if os.path.isabs(v) or v.startswith(("~", "\\")) or re.match(r"^[A-Za-z]:", v):
        return "an absolute path"
    parts = re.split(r"[\\/]+", v)
    if any(p == ".." for p in parts):
        return "a '..' component"
    return None


def _file_rules(xml_path):
    """The file-path rule on one MJCF file (the robot.xml / scene.xml of the submission; includes are separate files)."""
    v = []
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        return ["{} does not parse: {}".format(os.path.basename(xml_path), str(exc)[:160])]
    for el in root.iter():
        for k, val in el.attrib.items():
            if k == "file" or (el.tag == "compiler" and k in PATH_ATTRS):
                why = _bad_path(val)
                if why:
                    v.append("{}: <{} {}={!r}> is {} (every file must resolve inside the submission, whatever base MuJoCo "
                             "resolves it against)".format(os.path.basename(xml_path), el.tag, k, val, why))
    return v


def _header_rules(robot_xml, scene_xml):
    v = []
    try:
        rroot = ET.parse(robot_xml).getroot()
    except (ET.ParseError, OSError) as exc:
        return ["robot.xml does not parse: {}".format(str(exc)[:160])]
    comps = rroot.findall("compiler")
    if not comps or not any(c.attrib.get("angle") == "radian" for c in comps):
        v.append('robot.xml must state <compiler angle="radian"> explicitly (joint ranges and every angle in radians)')
    for c in comps:
        if c.attrib.get("angle") not in (None, "radian"):
            v.append("robot.xml: <compiler angle={}> (radian required)".format(c.attrib.get("angle")))
    if rroot.find(".//body[@name='base']") is None:
        v.append("robot.xml must define the robot subtree (body base)")
    with open(scene_xml, "rb") as f:
        stext = f.read().decode("utf-8", "replace")
    if not re.search(r'<include\s+file\s*=\s*"robot\.xml"\s*/?>', stext):
        v.append('scene.xml must <include file="robot.xml"/>')
    return v


def gate_model_loads(sub_dir, caps=None, whitelist_policy=None):
    """Returns (ok, details, models) with models = {"robot", "scene", "calibration"} (None where not loaded)."""
    import mujoco
    import numpy as np
    caps = dict(C.CAPS, **(caps or {}))
    v = []
    det = {"violations": v, "gate_version": GATE_VERSION}
    models = {"robot": None, "scene": None, "calibration": None}
    scene = os.path.join(sub_dir, "scene.xml")
    robot = os.path.join(sub_dir, "robot.xml")
    calp = os.path.join(sub_dir, "calibration.json")
    for nm, p in (("scene.xml", scene), ("robot.xml", robot), ("calibration.json", calp)):
        if not os.path.isfile(p) or os.path.islink(p):
            v.append("{} missing (a regular file in the submission root)".format(nm))
    if v:
        return False, det, models
    regular, irregular = submission_files(sub_dir)
    total_bytes = int(sum(regular.values()))
    det["submission_bytes"] = total_bytes
    det["submission_files"] = len(regular)
    if irregular:
        v.append("submission must contain regular files only: {}".format(irregular[:6]))
    if total_bytes > int(caps["max_submission_bytes"]):
        v.append("submission size {} bytes exceeds {}".format(total_bytes, caps["max_submission_bytes"]))
    try:
        from common.mjcf_whitelist import check_mjcf
    except ImportError:  # pragma: no cover
        from mjcf_whitelist import check_mjcf  # type: ignore
    for nm, p in (("scene.xml", scene), ("robot.xml", robot)):
        ok_w, viol, wstats = check_mjcf(p, submission_root=sub_dir, policy=whitelist_policy, robot_root="base")
        det["whitelist_stats_" + nm.split(".")[0]] = wstats
        if not ok_w:
            v += ["whitelist ({}): {}".format(nm, s) for s in viol[:10]]
    if v:
        return False, det, models             # nothing untrusted is compiled once a static rule failed
    v += _file_rules(scene) + _file_rules(robot)
    if v:
        return False, det, models
    v += _header_rules(robot, scene)
    cal, cv = load_calibration(calp)
    v += cv
    if v:
        return False, det, models
    models["calibration"] = cal
    for nm, p in (("scene", scene), ("robot", robot)):
        m, warnings, error = compile_model(p)
        det["compile_warnings_" + nm] = len(warnings)
        if error:
            v.append("{}.xml compile error: {}".format(nm, error[:300]))
            continue
        if warnings:
            v.append("{}.xml compile warnings: {}".format(nm, " | ".join(w[:160] for w in warnings[:3])))
        models[nm] = m
    if v:
        return False, det, models
    m = models["robot"]
    # --- structure (on robot.xml; scene.xml must carry the same robot)
    jn = _names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
    det["joints"] = jn
    if jn != C.JOINTS:
        v.append("joints {} != the contract's six hinges {} (in chain order, no other joint)".format(jn, C.JOINTS))
    else:
        for i in range(m.njnt):
            if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
                v.append("joint {} is not a hinge".format(jn[i]))
    if m.nq != 6 or m.nv != 6:
        v.append("nq / nv = {} / {} (six hinge DoF required)".format(m.nq, m.nv))
    if m.nq == 6 and float(np.max(np.abs(m.qpos0))) > 1e-12:
        v.append("a joint carries `ref` (qpos0 {}): q is the design angle, no ref (robot_contract.md)".format(np.round(m.qpos0, 6).tolist()))
    bn = _names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody)
    groups = body_groups(m)
    for b in C.BODIES:
        if b not in bn:
            v.append("body {} missing".format(b))
            continue
        i = bn.index(b)
        par = int(m.body_parentid[i])
        pg = groups[par] if par > 0 else "world"
        if pg != C.PARENT[b]:
            v.append("body {}: its nearest contract ancestor is {}, the contract says {}".format(b, pg, C.PARENT[b]))
        if b == "base":
            k = i
            while k > 0:
                if int(m.body_jntnum[k]) != 0:
                    v.append("body base (and its ancestors) must be welded to the world (no joint)")
                    break
                k = int(m.body_parentid[k])
    if not v:
        for j, b in C.BODY_OF_JOINT.items():
            ji = jn.index(j)
            if bn[int(m.jnt_bodyid[ji])] != b:
                v.append("joint {} must belong to body {} (it is on {})".format(j, b, bn[int(m.jnt_bodyid[ji])]))
        for i in range(m.nbody):
            if bn[i] not in C.BODIES and int(m.body_jntnum[i]) != 0:
                v.append("body {} carries a joint (only the contract bodies move)".format(bn[i]))
    for s, owner in C.SITES.items():
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        if sid < 0:
            v.append("site {} missing".format(s))
        elif groups[int(m.site_bodyid[sid])] != owner:
            v.append("site {} must be rigid on {} (it is on {})".format(s, owner, groups[int(m.site_bodyid[sid])]))
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home") < 0:
        v.append("keyframe home missing")
    an = _names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)
    det["actuators"] = an
    if sorted(an) != sorted(C.ACTUATORS):
        v.append("actuators {} != {}".format(an, sorted(C.ACTUATORS)))
    for a, j in C.ACTUATORS.items():
        ai = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if ai < 0:
            continue
        if int(m.actuator_trntype[ai]) != int(mujoco.mjtTrn.mjTRN_JOINT) or \
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[ai, 0])) != j:
            v.append("actuator {} must drive joint {}".format(a, j))
        if abs(float(m.actuator_gear[ai, 0]) - 1.0) > 1e-9:
            v.append("actuator {} gear must be 1".format(a))
        kp = float(m.actuator_gainprm[ai, 0])
        if int(m.actuator_gaintype[ai]) != int(mujoco.mjtGain.mjGAIN_FIXED) or \
                int(m.actuator_biastype[ai]) != int(mujoco.mjtBias.mjBIAS_AFFINE) or \
                abs(float(m.actuator_biasprm[ai, 1]) + kp) > 1e-9 * max(1.0, abs(kp)) or kp <= 0:
            v.append("actuator {} must be a position servo (<position> or an equivalent <general>)".format(a))
    # --- the scene carries the same robot (same joints / bodies)
    ms = models["scene"]
    if ms is not None:
        sj = _names(ms, mujoco.mjtObj.mjOBJ_JOINT, ms.njnt)
        if sj != jn:
            v.append("scene.xml joints {} differ from robot.xml {}".format(sj, jn))
    # --- caps
    faces = int(np.sum(m.mesh_facenum)) if m.nmesh else 0
    det["mesh_faces_total"] = faces
    if faces > int(caps["max_mesh_faces"]):
        v.append("mesh faces {} exceed {}".format(faces, caps["max_mesh_faces"]))
    coll = collision_geoms_by_group(m, groups)
    n_coll = sum(len(gs) for b, gs in coll.items() if b != "world")
    det["collision_geoms_total"] = n_coll
    det["collision_geoms_per_link"] = {b: len(gs) for b, gs in coll.items()}
    if n_coll > int(caps["max_collision_geoms_total"]):
        v.append("{} collision geoms exceed {}".format(n_coll, caps["max_collision_geoms_total"]))
    for b, gs in coll.items():
        if b != "world" and len(gs) > int(caps["max_collision_geoms_per_body"]):
            v.append("link {} has {} collision geoms (max {})".format(b, len(gs), caps["max_collision_geoms_per_body"]))
        for g in gs:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                fn = int(m.mesh_facenum[int(m.geom_dataid[g])])
                if fn > int(caps["max_collision_mesh_faces"]):
                    v.append("collision mesh of geom {} has {} faces (max {})".format(
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), fn, caps["max_collision_mesh_faces"]))
                    break
    return (not v), det, models
