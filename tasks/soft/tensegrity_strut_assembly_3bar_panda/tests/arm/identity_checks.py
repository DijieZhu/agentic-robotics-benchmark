"""Cross-stage identity checks C0, C-A, C1, C2, C3, C3', C4 (CHAIN_PLAN.md section 2.6).

Gates in the e2e flagship (each multiplies into every credited criterion), stats in the atomic
tasks where the given is the bank. Every function returns `(passed: bool, details: dict)`;
`details["violations"]` lists the reasons a check failed, `details` also carries the measured
numbers so a grader can log them as stats. Tolerances are the section-2.6 values as defaults.
Nothing here executes agent code: C2 / C4 verify the artefacts and the facts a grader collected
while it ran the agent's code itself (deprivileged).

`joint_zero_offset_rad` convention (CHAIN_PLAN 6 Q14, 2026-09-27; full text in `mjcf_reference.py`):
MuJoCo qpos == the design joint angle, robot.xml authors the design frames and carries no `ref`, the
offset of an as-built arm is servo calibration data. C1 therefore commands qpos = q and compares with
`Chain.fk(q)` (which ignores the offset) and rejects any `ref`; C-A verifies that an as-built A3's
offsets were NOT baked into robot.xml (child frame at qpos = 0 == design origin, no `ref`) and names the
legacy double compensation (`ref = -offset` plus `home` qpos = offset) explicitly.
"""

import ast
import math
import os
import xml.etree.ElementTree as ET

import numpy as np


from .chain_model import Chain, driven_by_list
from .frames import apply, inv, matrix_from_quat, pose_from_dict, rotation_angle_deg
from .geometry import sha256_file
from .handoff import ARM_JOINTS, JAW_JOINT, MM, actuator_facts_from_bom, load_json, tool_from_kit

__all__ = ["c0_requirements_to_hardware_design", "c_a_assembly_identity", "c1_description_matches_design",
           "c2_control_handoff", "c3_environment_robot_identity", "c3prime_mount_interface", "c4_behavior_wire",
           "verify_ik_solution"]


def _result(violations, **details):
    details["violations"] = list(violations)
    return (len(violations) == 0), details


# ============================================================================= C0
def c0_requirements_to_hardware_design(sketch_si, bom_si, assembly_mm, cad_bom, catalogue,
                                       joint_range_used_rad=None, tool=None, standin_map=None,
                                       shoulder_height_band_m=0.05, fact_tol=1e-9):
    """C0: the design honours the requirements output and copies the catalogue facts.

    Inputs: stage-1 `kinematic_sketch.json` (SI: base_height_m, links{link_k_m{min,max}},
    joints{name{servo_id,count}}), stage-1 `bom.json` (parts[] {part_id, count, role}), the design's
    `assembly.json` (mm), the CAD-facing `bom.json` (occurrences), `component_catalogue.json`,
    optional `joint_range_used_rad` {joint: [lo, hi]} (A1) that the declared limits must contain.
    Checks (2.6): per joint, occurrence count == sketch count and catalogue_id matches; copied
    facts (mass, stall, speed) equal the catalogue; link axis distances (shoulder->elbow,
    elbow->wrist, wrist->tcp) inside the sketch bounds; shoulder axis height in
    [base_height, base_height + 0.05 m]; declared limits contain joint_range_used_rad.
    """
    sketch = load_json(sketch_si)
    bom1 = load_json(bom_si)
    asm = load_json(assembly_mm)
    cad_bom = load_json(cad_bom)
    cat = load_json(catalogue)
    facts = actuator_facts_from_bom(cad_bom)
    cat_by_id = {}
    for group in ("servos", "bus_boards", "psus"):
        for item in cat.get(group, []):
            cat_by_id[item["id"]] = item
    chain = Chain(asm, tool or tool_from_kit(None))
    v = []
    details = {"per_joint": {}}
    # occurrence counts and catalogue ids per joint
    sketch_joints = sketch.get("joints", {})
    for j in chain.joints:
        want = sketch_joints.get(j.name, {})
        ids = driven_by_list(j.spec["driven_by"])
        rec = {"occurrences": ids, "sketch_count": want.get("count"), "sketch_servo": want.get("servo_id")}
        if want and len(ids) != int(want.get("count", 0)):
            v.append("{}: {} driving occurrence(s), sketch says {}".format(j.name, len(ids), want.get("count")))
        for sid in ids:
            f = facts.get(sid)
            if f is None:
                v.append("{}: occurrence {} missing from the CAD bom".format(j.name, sid))
                continue
            if want and f["catalogue_id"] != want.get("servo_id"):
                v.append("{}: {} is {} but the sketch says {}".format(j.name, sid, f["catalogue_id"], want.get("servo_id")))
        details["per_joint"][j.name] = rec
    # the gripper servo
    grip = sketch_joints.get("gripper")
    if grip:
        g_ids = [sid for sid, _, _ in chain.all_occurrences() if sid not in sum([driven_by_list(j.spec["driven_by"]) for j in chain.joints], [])]
        if len(g_ids) != int(grip.get("count", 1)):
            v.append("gripper: {} carried occurrence(s) {}, sketch says {}".format(len(g_ids), g_ids, grip.get("count")))
        for sid in g_ids:
            if sid in facts and facts[sid]["catalogue_id"] != grip.get("servo_id"):
                v.append("gripper: {} is {} but the sketch says {}".format(sid, facts[sid]["catalogue_id"], grip.get("servo_id")))
    # copied facts equal the catalogue
    for sid, f in facts.items():
        c = cat_by_id.get(f["catalogue_id"])
        if c is None:
            v.append("{}: catalogue_id {} is not in the catalogue".format(sid, f["catalogue_id"]))
            continue
        for key in ("mass_kg", "stall_torque_nm", "no_load_speed_rad_s"):
            if f.get(key) is None:
                v.append("{}: {} not copied".format(sid, key))
            elif abs(float(f[key]) - float(c[key])) > fact_tol:
                v.append("{}: {} {} differs from the catalogue {}".format(sid, key, f[key], c[key]))
    # stage-1 bom counts vs occurrences per SKU
    counts = {}
    for f in facts.values():
        counts[f["catalogue_id"]] = counts.get(f["catalogue_id"], 0) + 1
    for part in bom1.get("parts", []):
        if part.get("role") == "servo" and counts.get(part["part_id"], 0) != int(part["count"]):
            v.append("bom: {} x{} in the requirements bom, {} occurrence(s) in the design".format(
                part["part_id"], part["count"], counts.get(part["part_id"], 0)))
    # geometry: axis distances and shoulder height (mm -> m)
    T0 = chain.fk(np.zeros(len(chain.joints)))
    def axis(name):
        j = chain.by_name[name]
        return chain.axis_line_world(j, T0)
    def line_dist(pa, da, pb, db):
        n = np.cross(da, db)
        if np.linalg.norm(n) < 1e-9:
            d = pb - pa
            return float(np.linalg.norm(d - (d @ da) * da))
        return abs(float((pb - pa) @ n)) / float(np.linalg.norm(n))
    dists = {}
    if all(n in chain.by_name for n in ("shoulder_pitch", "elbow_pitch", "wrist_pitch")):
        ps, ds = axis("shoulder_pitch")
        pe, de = axis("elbow_pitch")
        pw, dw = axis("wrist_pitch")
        dists["link_1_m"] = line_dist(ps, ds, pe, de) * MM
        dists["link_2_m"] = line_dist(pe, de, pw, dw) * MM
        if chain.tool and chain.gripper_occurrence()[0]:
            tcp, _ = chain.tcp(np.zeros(len(chain.joints)), T0)
            d = tcp - pw
            dists["link_3_m"] = float(np.linalg.norm(d - (d @ dw) * dw)) * MM
        dists["shoulder_axis_height_m"] = float(ps[2]) * MM
    links = sketch.get("links", {})
    for key in ("link_1_m", "link_2_m", "link_3_m"):
        if key in dists and key in links:
            lo, hi = float(links[key]["min"]), float(links[key]["max"])
            if not (lo - 1e-6 <= dists[key] <= hi + 1e-6):
                v.append("{} = {:.4f} m outside the sketch bounds [{}, {}]".format(key, dists[key], lo, hi))
    if "shoulder_axis_height_m" in dists and "base_height_m" in sketch:
        h0 = float(sketch["base_height_m"])
        h = dists["shoulder_axis_height_m"]
        if not (h0 - 1e-6 <= h <= h0 + shoulder_height_band_m + 1e-6):
            v.append("shoulder axis height {:.4f} m outside [{}, {}]".format(h, h0, h0 + shoulder_height_band_m))
    details["distances_m"] = dists
    # declared limits contain the used ranges
    if joint_range_used_rad:
        for j in chain.joints:
            rng = joint_range_used_rad.get(j.name)
            if rng and not (j.lower <= float(rng[0]) + 1e-9 and j.upper >= float(rng[1]) - 1e-9):
                v.append("{} limits [{}, {}] do not contain joint_range_used_rad {}".format(j.name, j.lower, j.upper, rng))
    return _result(v, **details)


# ============================================================================= C-A
def c_a_assembly_identity(design_dir, as_built_dir, assembly_record, robot_xml=None,
                          tol_pos_mm=1.5, tol_ang_deg=3.0, offset_tol=1e-6, origin_tol_m=1e-6, origin_tol_deg=1e-4):
    """C-A: assembly changed nothing but the recorded servo zeros, and robot.xml is still the design.

    `as_built/parts/` byte-identical to `hardware-design/parts/`; every achieved mate of
    `assembly_record.json` has pre-snap error within tol_pos / tol_ang; every horn mate records a
    yaw quadrant in {0, 90, 180, 270}; every as-built joint frame equals the design origin
    (`as_built_origin == origin` within origin_tol: the welds snap to the design, the yaw quadrant is the
    only as-built degree of freedom). When `robot_xml` is given, each joint whose
    `joint_zero_offset_rad` (as_built/assembly.json) is nonzero must NOT have been baked into the
    kinematics (convention Q14, `mjcf_reference.py`): the compiled child frame at qpos = 0 equals the
    design origin and the joint carries no `ref` (`qpos0 == 0`), so qpos is the design angle and the
    offset stays calibration data (servo command = sign x (qpos - offset)). The legacy encodings are
    rejected with an explicit reason: `ref = -offset` on the design frame (qpos = 0 is then the
    servo-zero pose) and, when the `home` keyframe repeats the offset too, "double compensation".
    `details["robot_xml_offset_encoding"]` reports ref / home / zero-pose error per offset joint.
    """
    rec = load_json(assembly_record)
    v = []
    parts_a = os.path.join(design_dir, "parts")
    parts_b = os.path.join(as_built_dir, "parts")
    names_a = sorted(os.listdir(parts_a)) if os.path.isdir(parts_a) else []
    names_b = sorted(os.listdir(parts_b)) if os.path.isdir(parts_b) else []
    if names_a != names_b:
        v.append("as_built/parts/ lists {} vs hardware-design/parts/ {}".format(names_b, names_a))
    for n in names_a:
        if n in names_b and sha256_file(os.path.join(parts_a, n)) != sha256_file(os.path.join(parts_b, n)):
            v.append("as_built/parts/{} differs from hardware-design/parts/{}".format(n, n))
    achieved, quadrants = 0, {}
    worst = {"pos_err_mm": 0.0, "ang_err_deg": 0.0}
    for m in rec.get("mates", []):
        if not m.get("achieved"):
            continue
        achieved += 1
        pe, ae = float(m.get("pos_err_mm", 0.0)), float(m.get("ang_err_deg", 0.0))
        worst["pos_err_mm"] = max(worst["pos_err_mm"], pe)
        worst["ang_err_deg"] = max(worst["ang_err_deg"], ae)
        if pe > tol_pos_mm + 1e-9 or ae > tol_ang_deg + 1e-9:
            v.append("mate {} achieved with pre-snap error {:.3f} mm / {:.3f} deg beyond {} / {}".format(m["id"], pe, ae, tol_pos_mm, tol_ang_deg))
        if m.get("kind") == "horn_mate" or str(m.get("id", "")).startswith("horn_"):
            q = m.get("yaw_quadrant_deg")
            if q is None:
                v.append("horn mate {} has no yaw_quadrant_deg".format(m["id"]))
            elif int(round(float(q))) % 90 != 0:
                v.append("horn mate {} yaw quadrant {} is not a multiple of 90 deg".format(m["id"], q))
            quadrants[m["id"]] = q
    offsets, origin_dev, joints_b, scale_b = {}, {}, [], MM
    asm_b = os.path.join(as_built_dir, "assembly.json")
    asm_d = os.path.join(design_dir, "assembly.json")
    design_origins = {}
    if os.path.isfile(asm_d):
        design_origins = dict((j["name"], j["origin"]) for j in load_json(asm_d).get("joints", []))
    if os.path.isfile(asm_b):
        doc_b = load_json(asm_b)
        joints_b = doc_b.get("joints", [])
        scale_b = _length_scale(doc_b)
        for j in joints_b:
            offsets[j["name"]] = float(j.get("joint_zero_offset_rad", 0.0))
            Td = pose_from_dict(j["origin"])
            if j["name"] in design_origins:
                Tdd = pose_from_dict(design_origins[j["name"]])
                dp = float(np.linalg.norm(Tdd[:3, 3] - Td[:3, 3])) * scale_b
                da = rotation_angle_deg(Tdd[:3, :3], Td[:3, :3])
                if dp > origin_tol_m or da > origin_tol_deg:
                    v.append("joint {} origin in as_built/assembly.json deviates {:.3f} mm / {:.4f} deg from hardware-design/assembly.json".format(j["name"], dp * 1e3, da))
            if "as_built_origin" in j:
                Ta = pose_from_dict(j["as_built_origin"])
                dp = float(np.linalg.norm(Ta[:3, 3] - Td[:3, 3])) * scale_b
                da = rotation_angle_deg(Ta[:3, :3], Td[:3, :3])
                origin_dev[j["name"]] = {"pos_m": dp, "ang_deg": da}
                if dp > origin_tol_m or da > origin_tol_deg:
                    v.append("joint {} as_built_origin deviates {:.3f} mm / {:.4f} deg from the design origin (welds snap to the design; "
                             "the yaw quadrant is the only as-built degree of freedom)".format(j["name"], dp * 1e3, da))
    encoding = {}
    if robot_xml and offsets:
        encoding = _zero_offset_encoding(robot_xml, joints_b, scale_b)
        for name, off in offsets.items():
            if abs(off) < offset_tol:
                continue
            e = encoding.get(name)
            if e is None:
                v.append("robot.xml has no joint {} (as-built zero offset {:+.6f} rad)".format(name, off))
                continue
            baked = e["zero_pose_err_m"] > origin_tol_m or e["zero_pose_err_deg"] > origin_tol_deg
            if baked or abs(e["ref"]) > offset_tol:
                why = ("at qpos = 0 the child frame is {:.4f} deg / {:.3f} mm off the design origin".format(e["zero_pose_err_deg"], e["zero_pose_err_m"] * 1e3)
                       if baked else "the authored frame was rotated to absorb the ref: an equivalent but non-conforming encoding")
                reason = ("joint {} zero offset {:+.6f} rad was baked into robot.xml (ref={:+.6f}: {}). Convention Q14: qpos == design angle, "
                          "the authored frame is the design origin, no ref; the offset is calibration data (servo command = sign x (qpos - offset))"
                          ).format(name, off, e["ref"], why)
                if e["home"] is not None and abs(abs(e["home"]) - abs(off)) <= max(offset_tol, 1e-5):
                    reason += "; the home keyframe repeats the offset (qpos {:+.6f}): the legacy double compensation".format(e["home"])
                v.append(reason)
    return _result(v, parts_checked=len(names_a), mates_achieved=achieved, worst_pre_snap=worst,
                   yaw_quadrants_deg=quadrants, joint_zero_offsets_rad=offsets, as_built_origin_deviation=origin_dev,
                   robot_xml_offset_encoding=encoding)


def _length_scale(doc):
    """Metres per length unit of an assembly.json (`units.length`: mm (the stage 2-3 default) or m)."""
    unit = str((doc.get("units") or {}).get("length", "mm")).lower()
    return 1.0 if unit in ("m", "metre", "meter", "metres", "meters") else MM


def _zero_offset_encoding(robot_xml, joints_spec, scale):
    """Per joint of `joints_spec` present in robot.xml: `ref` (qpos0), the `home` keyframe value and the
    error of the compiled child frame at qpos = 0 against the design origin (scaled to metres)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(robot_xml)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_kinematics(model, data)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    out = {}
    for j in joints_spec:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        b = int(model.jnt_bodyid[jid])
        p = int(model.body_parentid[b])
        T_b = np.eye(4)
        T_b[:3, :3] = data.xmat[b].reshape(3, 3)
        T_b[:3, 3] = data.xpos[b]
        T_p = np.eye(4)
        T_p[:3, :3] = data.xmat[p].reshape(3, 3)
        T_p[:3, 3] = data.xpos[p]
        rel = inv(T_p) @ T_b
        Td = pose_from_dict(j["origin"])
        Td[:3, 3] *= scale
        out[j["name"]] = {"ref": float(model.qpos0[qadr]),
                          "home": float(model.key_qpos[kid][qadr]) if kid >= 0 else None,
                          "zero_pose_err_m": float(np.linalg.norm(rel[:3, 3] - Td[:3, 3])),
                          "zero_pose_err_deg": rotation_angle_deg(rel[:3, :3], Td[:3, :3])}
    return out


# ============================================================================= C1
def c1_description_matches_design(assembly_si, robot_xml, actuator_facts=None, n_configs=200, seed=0,
                                  tol_pos_m=1e-3, tol_ang_deg=0.5, mass_tol=0.05, range_tol=1e-6, axis_tol_deg=1e-3,
                                  ref_tol=1e-9):
    """C1: robot.xml is the design (A3 output) it claims to be.

    Joint count / order / axes / ranges equal; no joint carries `ref` (qpos == design angle, Q14);
    FK of every body frame and the `tcp` site at `n_configs` configurations (numpy PCG64(seed): the
    caller passes a hidden seed), commanded as qpos = q and compared with `Chain.fk(q)` (which ignores
    `joint_zero_offset_rad`: an as-built A3 must yield the same FK as the design), within tol_pos /
    tol_ang of the assembly.json + gripper chain; `act_<joint>` forcerange within +-count x stall
    (actuator_facts.json); body masses within mass_tol of mass_properties.
    """
    import mujoco
    asm = load_json(assembly_si)
    facts = load_json(actuator_facts)["joints"] if actuator_facts else None
    chain = Chain(asm)
    v = []
    model = mujoco.MjModel.from_xml_path(robot_xml)
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    want = [j.name for j in chain.joints] + [asm["gripper"]["joint"]["name"]]
    if names[:len(want)] != want or len(names) != len(want):
        v.append("joint list {} != {}".format(names, want))
        return _result(v)
    # convention Q14: qpos is the design joint angle, so no joint may carry `ref` (MuJoCo rotates the child by
    # qpos - ref; the pre-Q14 writer's ref = -offset made qpos = 0 the servo-zero pose, bank row C1')
    joint_refs = dict((names[i], float(model.qpos0[int(model.jnt_qposadr[i])])) for i in range(model.njnt))
    for name, r in joint_refs.items():
        if abs(r) > ref_tol:
            v.append("joint {} has ref={:+.6f}: robot.xml qpos must be the design joint angle (no ref; a servo-zero "
                     "offset is calibration data, CHAIN_PLAN Q14)".format(name, r))
    # ranges and axes (child-frame axis; compare in the world frame at q = 0 to be frame-agnostic)
    T0 = chain.fk(np.zeros(len(chain.joints)))
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0
    mujoco.mj_kinematics(model, data)
    g = asm["gripper"]
    carrier = g["carrier_link"]
    T_jaw0 = T0[carrier] @ pose_from_dict(g["joint"]["origin"])
    design_axes = dict((j.name, T0[j.child][:3, :3] @ j.axis) for j in chain.joints)
    design_axes[g["joint"]["name"]] = T_jaw0[:3, :3] @ np.asarray(g["joint"]["axis"], float)
    design_ranges = dict((j.name, (j.lower, j.upper)) for j in chain.joints)
    design_ranges[g["joint"]["name"]] = (float(g["joint"]["limits"]["lower"]), float(g["joint"]["limits"]["upper"]))
    for i, name in enumerate(names):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            v.append("joint {} is not a hinge".format(name))
        lo, hi = model.jnt_range[i]
        dlo, dhi = design_ranges[name]
        if abs(lo - dlo) > range_tol or abs(hi - dhi) > range_tol:
            v.append("joint {} range [{}, {}] != design [{}, {}]".format(name, lo, hi, dlo, dhi))
        ax_w = data.xmat[model.jnt_bodyid[i]].reshape(3, 3) @ model.jnt_axis[i]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(ax_w @ design_axes[name]) / (np.linalg.norm(ax_w) * np.linalg.norm(design_axes[name]))))))
        if ang > axis_tol_deg:
            v.append("joint {} axis deviates {:.4f} deg from the design".format(name, ang))
    # FK identity at hidden configurations
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    qadr = [int(model.jnt_qposadr[i]) for i in range(model.njnt)]
    lo = np.array([r[0] for r in (design_ranges[n] for n in names)])
    hi = np.array([r[1] for r in (design_ranges[n] for n in names)])
    bodies = [n for n in chain.order] + ["jaw_moving"]
    bid = {}
    for b in bodies:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if i < 0:
            v.append("robot.xml has no body {}".format(b))
        bid[b] = i
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if sid < 0:
        v.append("robot.xml has no site tcp")
    if v:
        return _result(v)
    T_tcp_local = pose_from_dict({"xyz": asm["tcp"]["xyz"], "rpy": asm["tcp"]["rpy"]})
    T_jaw_origin = pose_from_dict(g["joint"]["origin"])
    jaw_axis = np.asarray(g["joint"]["axis"], float)
    worst_p, worst_a, worst_site = 0.0, 0.0, 0.0
    from .frames import rot_axis
    for _ in range(int(n_configs)):
        q = lo + rng.random(len(names)) * (hi - lo)
        # q are design joint angles == qpos (no ref, Q14); Chain.fk ignores any joint_zero_offset_rad
        data.qpos[:] = 0
        for i, name in enumerate(names):
            data.qpos[qadr[i]] = q[i]
        mujoco.mj_kinematics(model, data)
        Td = chain.fk(q[:len(chain.joints)])
        Td["jaw_moving"] = Td[carrier] @ T_jaw_origin @ rot_axis(jaw_axis, q[-1])
        for b in bodies:
            p_mj = data.xpos[bid[b]]
            R_mj = data.xmat[bid[b]].reshape(3, 3)
            worst_p = max(worst_p, float(np.linalg.norm(p_mj - Td[b][:3, 3])))
            worst_a = max(worst_a, rotation_angle_deg(R_mj, Td[b][:3, :3]))
        T_tcp = Td[carrier] @ T_tcp_local
        worst_site = max(worst_site, float(np.linalg.norm(data.site_xpos[sid] - T_tcp[:3, 3])))
        worst_a = max(worst_a, rotation_angle_deg(data.site_xmat[sid].reshape(3, 3), T_tcp[:3, :3]))
    if worst_p > tol_pos_m or worst_site > tol_pos_m:
        v.append("FK position deviation {:.5f} m (bodies) / {:.5f} m (tcp) exceeds {} m".format(worst_p, worst_site, tol_pos_m))
    if worst_a > tol_ang_deg:
        v.append("FK orientation deviation {:.4f} deg exceeds {} deg".format(worst_a, tol_ang_deg))
    # actuators
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    force_report = {}
    for name in names:
        an = "act_{}".format(name)
        if an not in act_names:
            v.append("no actuator {}".format(an))
            continue
        i = act_names.index(an)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])) != name:
            v.append("{} does not drive joint {}".format(an, name))
        cr = model.actuator_ctrlrange[i]
        if abs(cr[0] - design_ranges[name][0]) > 1e-3 or abs(cr[1] - design_ranges[name][1]) > 1e-3:
            v.append("{} ctrlrange {} != joint range {}".format(an, cr.tolist(), design_ranges[name]))
        if facts and name in facts:
            cap = float(facts[name]["stall_torque_nm_total"])
            fr = model.actuator_forcerange[i]
            force_report[an] = {"forcerange": fr.tolist(), "count_x_stall": cap}
            if not model.actuator_forcelimited[i] or fr[0] < -cap - 1e-9 or fr[1] > cap + 1e-9:
                v.append("{} forcerange {} outside +-{} (count x stall)".format(an, fr.tolist(), cap))
    # inertials
    mass_report = {}
    for link in asm["links"]:
        i = bid[link["name"]]
        m_des = float(link["mass_properties"]["mass_kg"])
        m_mj = float(model.body_mass[i])
        rel = abs(m_mj - m_des) / max(m_des, 1e-9)
        com_err = float(np.linalg.norm(model.body_ipos[i] - np.asarray(link["mass_properties"]["com_m"])))
        mass_report[link["name"]] = {"mass_mj": m_mj, "mass_design": m_des, "rel_err": rel, "com_err_m": com_err}
        if rel > mass_tol:
            v.append("body {} mass {:.4f} kg deviates {:.1%} from the design {:.4f} kg".format(link["name"], m_mj, rel, m_des))
    m_des = float(g["moving_jaw"]["mass_properties"]["mass_kg"])
    m_mj = float(model.body_mass[bid["jaw_moving"]])
    mass_report["jaw_moving"] = {"mass_mj": m_mj, "mass_design": m_des, "rel_err": abs(m_mj - m_des) / max(m_des, 1e-9)}
    if mass_report["jaw_moving"]["rel_err"] > mass_tol:
        v.append("body jaw_moving mass {:.5f} kg deviates from the kit {:.5f} kg".format(m_mj, m_des))
    return _result(v, fk_worst_pos_m=worst_p, fk_worst_tcp_pos_m=worst_site, fk_worst_ang_deg=worst_a,
                   n_configs=int(n_configs), forcerange=force_report, masses=mass_report, joint_refs=joint_refs)


# ============================================================================= C2
_FORBIDDEN_IMPORTS = {"socket", "urllib", "requests", "http", "subprocess", "ftplib", "smtplib", "xmlrpc", "multiprocessing"}
_MODEL_LOADERS = {"from_xml_path", "from_xml_string", "from_binary_path", "mj_loadXML", "mj_loadModel"}


def c2_control_handoff(robot_xml, mjb_out, control_dir=None):
    """C2: the .mjb hand-off and the static half of the control contract.

    The grader compiles the submitted robot.xml, writes `.mjb`, and the child process loads it
    read-only (05 A9): compile -> save -> reload must round-trip (same nq, nu, names, ranges).
    When `control_dir` is given, `ik.py` (and sibling modules) are audited by AST: no network /
    subprocess imports, no model loading of its own (`MjModel.from_*`: ik.py may not load any
    other model), file I/O calls reported as a stat. Solutions are verified by the grader's FK
    at rollout time (`verify_ik_solution`); this function does not run agent code.
    """
    import mujoco
    v = []
    model = mujoco.MjModel.from_xml_path(robot_xml)
    os.makedirs(os.path.dirname(os.path.abspath(mjb_out)), exist_ok=True)
    mujoco.mj_saveModel(model, mjb_out, None)
    reloaded = mujoco.MjModel.from_binary_path(mjb_out)
    sig = lambda m: (int(m.nq), int(m.nu), int(m.nbody), [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)],
                     np.round(m.jnt_range, 9).tolist())
    if sig(model) != sig(reloaded):
        v.append(".mjb round-trip changed the model signature")
    audit = {"files": [], "forbidden_imports": [], "model_loads": [], "file_io_calls": 0}
    if control_dir:
        for name in sorted(os.listdir(control_dir)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(control_dir, name)
            audit["files"].append(name)
            try:
                tree = ast.parse(open(path).read(), filename=name)
            except SyntaxError as exc:
                v.append("{}: syntax error {}".format(name, exc))
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    for m in mods:
                        if m.split(".")[0] in _FORBIDDEN_IMPORTS:
                            audit["forbidden_imports"].append("{}:{} {}".format(name, node.lineno, m))
                if isinstance(node, ast.Attribute) and node.attr in _MODEL_LOADERS:
                    audit["model_loads"].append("{}:{} .{}".format(name, node.lineno, node.attr))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                    audit["file_io_calls"] += 1
        if not any(f == "ik.py" for f in audit["files"]):
            v.append("control/ has no ik.py")
        if audit["forbidden_imports"]:
            v.append("forbidden imports in control/: {}".format(audit["forbidden_imports"][:5]))
        if audit["model_loads"]:
            v.append("ik.py loads a model of its own (C2 passes the grader's model): {}".format(audit["model_loads"][:5]))
    return _result(v, mjb=mjb_out, mjb_sha256=sha256_file(mjb_out), nq=int(model.nq), nu=int(model.nu), audit=audit)


def verify_ik_solution(model, data, q, target, tol_pos_m=0.002, tol_axis_deg=1.0, limit_slack=1e-6):
    """Grader-side FK verification of one IK answer (the C2 rule 'solutions verified by grader FK')."""
    import mujoco
    q = np.asarray(q, float).ravel()
    if q.shape != (5,) or not np.all(np.isfinite(q)):
        return False, {"reason": "not a finite (5,) vector"}
    idx = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
    rng = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])
    if np.any(q < rng[:, 0] - limit_slack) or np.any(q > rng[:, 1] + limit_slack):
        return False, {"reason": "outside joint limits"}
    data.qpos[idx] = q
    mujoco.mj_kinematics(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    pos = data.site_xpos[sid]
    z = data.site_xmat[sid].reshape(3, 3)[:, 2]
    perr = float(np.linalg.norm(pos - np.asarray(target["pos"], float)))
    a = np.asarray(target["approach"], float)
    aerr = math.degrees(math.acos(max(-1.0, min(1.0, float(z @ a) / np.linalg.norm(a)))))
    ok = perr <= tol_pos_m and aerr <= tol_axis_deg
    if ok and target.get("quat") is not None:
        ok = rotation_angle_deg(data.site_xmat[sid].reshape(3, 3), matrix_from_quat(target["quat"])) <= tol_axis_deg
    return ok, {"pos_err_m": perr, "axis_err_deg": aerr}


# ============================================================================= C3
def c3_environment_robot_identity(robot_xml, scene_xml, mount_pos_m, mount_quat_wxyz=(1, 0, 0, 0), robot_root="base",
                                  submission_root=None, task_objects=("mug_1", "sink_1"), asset_lock=None,
                                  mount_tol_m=1e-6, mount_tol_deg=1e-4, whitelist_policy=None):
    """C3: the environment's scene.xml contains the submitted robot unchanged, at the mount frame.

    Canonicalised robot subtree equal (1e-9), actuators unchanged, root body placed at
    (mount_pos_m, mount_quat_wxyz), the F1 whitelist holds for scene.xml, and every mesh the
    task objects (`mug_1`, `sink_1`) reference hashes to an entry of `asset_lock` ({name or path:
    sha256} or [{path, sha256}]); without an asset lock that sub-check is reported as skipped.
    """
    from common.mjcf_canonical import canonical_robot, canonical_sha256, find_body, referenced_files
    from common.mjcf_whitelist import check_mjcf
    v = []
    try:
        cr = canonical_robot(robot_xml, robot_root)
    except ValueError as exc:
        return _result(["robot.xml: {}".format(exc)])
    try:
        cs = canonical_robot(scene_xml, robot_root)
    except ValueError as exc:
        return _result(["scene.xml: {}".format(exc)])
    if cr["subtree"] != cs["subtree"]:
        v.append("robot subtree in scene.xml differs from robot.xml after canonicalisation")
    if cr["actuators"] != cs["actuators"]:
        v.append("actuators of the robot joints differ between robot.xml and scene.xml")
    if cr["meshes"] != cs["meshes"]:
        v.append("robot mesh assets differ between robot.xml and scene.xml")
    # mount frame
    pl = cs["placement"]
    pos = [float(x) for x in pl.get("pos", "0 0 0").split()]
    if "quat" in pl:
        quat = [float(x) for x in pl["quat"].split()]
    elif any(k in pl for k in ("euler", "axisangle", "xyaxes", "zaxis")):
        quat = None
        v.append("robot root body uses {} for its orientation; state it as quat".format([k for k in pl if k != "pos"]))
    else:
        quat = [1.0, 0.0, 0.0, 0.0]
    perr = float(np.linalg.norm(np.asarray(pos) - np.asarray(mount_pos_m, float)))
    if perr > mount_tol_m:
        v.append("robot root at {} is {:.6f} m from the mount frame {}".format(pos, perr, list(mount_pos_m)))
    aerr = None
    if quat is not None:
        aerr = rotation_angle_deg(matrix_from_quat(quat), matrix_from_quat(mount_quat_wxyz))
        if aerr > mount_tol_deg:
            v.append("robot root orientation deviates {:.6f} deg from the mount frame".format(aerr))
    ok_w, viol, stats = check_mjcf(scene_xml, submission_root=submission_root or os.path.dirname(os.path.abspath(scene_xml)),
                                   policy=whitelist_policy, robot_root=robot_root)
    if not ok_w:
        v.append("whitelist: " + "; ".join(viol[:8]))
    # asset identity of the task objects
    asset_report = {"checked": [], "skipped": asset_lock is None}
    if asset_lock is not None:
        lock = {}
        entries = asset_lock if isinstance(asset_lock, list) else [{"path": k, "sha256": s} for k, s in asset_lock.items()]
        for e in entries:
            lock[os.path.basename(e["path"])] = e["sha256"]
        root = ET.parse(scene_xml).getroot()
        meshes = dict((m.attrib.get("name"), m.attrib.get("file")) for a in root.iter("asset") for m in a if m.tag == "mesh")
        files = dict((os.path.basename(p), p) for p in referenced_files(scene_xml))
        for name in task_objects:
            body = find_body(root, name)
            if body is None:
                v.append("scene.xml has no body {}".format(name))
                continue
            for geom in body.iter("geom"):
                mesh = geom.attrib.get("mesh")
                if not mesh:
                    continue
                fname = os.path.basename(meshes.get(mesh, ""))
                path = files.get(fname)
                digest = sha256_file(path) if path and os.path.isfile(path) else None
                asset_report["checked"].append({"body": name, "mesh": mesh, "file": fname, "sha256": digest})
                if fname not in lock or lock[fname] != digest:
                    v.append("{}: mesh {} ({}) is not a library asset (sha256 mismatch)".format(name, mesh, fname))
    return _result(v, robot_canonical_sha256=canonical_sha256(cr), scene_robot_canonical_sha256=canonical_sha256(cs),
                   mount_pos_err_m=perr, mount_ang_err_deg=aerr, whitelist_stats=stats, assets=asset_report)


# ============================================================================= C3'
def _placement_rotation(elem, compiler_angle="radian"):
    """Rotation matrix of a body / geom placement: `quat` (default identity), `euler` (compiler angle unit, XYZ intrinsic = MuJoCo's default
    eulerseq) or `axisangle`; `xyaxes` / `zaxis` are reported as unsupported (None)."""
    if "quat" in elem.attrib:
        return matrix_from_quat([float(x) for x in elem.attrib["quat"].split()])
    if "euler" in elem.attrib:
        e = np.array([float(x) for x in elem.attrib["euler"].split()], float)
        if compiler_angle == "degree":
            e = np.radians(e)
        R = np.eye(3)
        for ang, axis in zip(e, ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            R = R @ _axis_angle(axis, ang)
        return R
    if "axisangle" in elem.attrib:
        a = [float(x) for x in elem.attrib["axisangle"].split()]
        ang = a[3] if compiler_angle == "radian" else np.radians(a[3])
        return _axis_angle(a[:3], ang)
    if "xyaxes" in elem.attrib or "zaxis" in elem.attrib:
        return None
    return np.eye(3)


def _axis_angle(axis, ang):
    k = np.asarray(axis, float)
    k = k / max(1e-12, np.linalg.norm(k))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def c3prime_mount_interface(robot_xml, robot_root="base", max_footprint_mm=(110.0, 110.0), max_height_mm=60.0, floor_tol_mm=0.5):
    """C3': the base link fits the reference kitchen's mount interface.

    AABB of every geom of the root body AND of every descendant body welded to it (a child body that carries no `joint` /
    `freejoint` is part of the static base; the walk stops at the first body with a joint, i.e. the first moving link), meshes
    loaded from meshdir, primitives by size, body placements composed: footprint <= 110 x 110 mm, top <= 60 mm above the
    mounting plane z = 0, bottom on the plane. (Until 2026-09-27 only the direct geoms of the root body were measured, so a
    plate in a jointless child body escaped the gate: LENS A review of the e2e flagship.)
    """
    import trimesh
    root = ET.parse(robot_xml).getroot()
    from common.mjcf_canonical import find_body
    body = find_body(root, robot_root)
    if body is None:
        return _result(["no body {}".format(robot_root)])
    comp = root.find("compiler")
    meshdir = comp.attrib.get("meshdir", "") if comp is not None else ""
    angle_unit = comp.attrib.get("angle", "degree") if comp is not None else "degree"
    base_dir = os.path.dirname(os.path.abspath(robot_xml))
    meshes = dict((m.attrib.get("name"), m) for a in root.iter("asset") for m in a if m.tag == "mesh")
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n = 0
    welded = []
    v = []
    stack = [(body, np.eye(3), np.zeros(3))]
    while stack:
        b, Rb, pb = stack.pop()
        for child in list(b):
            if child.tag == "body":
                if any(gc.tag in ("joint", "freejoint") for gc in list(child)):
                    continue                                  # the first moving link: not part of the static base
                Rc = _placement_rotation(child, angle_unit)
                pc = np.array([float(x) for x in child.attrib.get("pos", "0 0 0").split()])
                if Rc is None:
                    v.append("welded body {} uses xyaxes / zaxis for its orientation; state it as quat".format(child.attrib.get("name")))
                    continue
                welded.append(child.attrib.get("name"))
                stack.append((child, Rb @ Rc, pb + Rb @ pc))
                continue
            if child.tag != "geom":
                continue
            geom = child
            gpos = np.array([float(x) for x in geom.attrib.get("pos", "0 0 0").split()])
            R = _placement_rotation(geom, angle_unit)
            if R is None:
                v.append("geom {} uses xyaxes / zaxis for its orientation; state it as quat".format(geom.attrib.get("name")))
                continue
            gtype = geom.attrib.get("type", "mesh" if "mesh" in geom.attrib else "sphere")
            if gtype == "mesh":
                m = meshes.get(geom.attrib.get("mesh"))
                if m is None:
                    continue
                scale = np.array([float(x) for x in m.attrib.get("scale", "1 1 1").split()])
                tm = trimesh.load(os.path.join(base_dir, meshdir, m.attrib["file"]), force="mesh")
                pts = (np.asarray(tm.vertices) * scale) @ R.T + gpos
            elif gtype in ("box", "cylinder", "capsule", "sphere", "ellipsoid"):
                s = np.array([float(x) for x in geom.attrib.get("size", "0").split()])
                half = {"box": s[:3] if len(s) >= 3 else np.repeat(s[0], 3),
                        "cylinder": np.array([s[0], s[0], s[1]]), "capsule": np.array([s[0], s[0], s[1] + s[0]]),
                        "sphere": np.repeat(s[0], 3), "ellipsoid": s[:3]}[gtype]
                corners = np.array([[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                pts = corners @ R.T + gpos
            else:
                continue
            pts = pts @ Rb.T + pb                                 # into the root body's frame
            n += 1
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
    if n == 0:
        return _result(v + ["root body {} has no measurable geoms".format(robot_root)])
    ext_mm = (hi - lo) * 1e3
    if ext_mm[0] > max_footprint_mm[0] + 1e-6 or ext_mm[1] > max_footprint_mm[1] + 1e-6:
        v.append("base footprint {:.1f} x {:.1f} mm exceeds {} x {}{}".format(ext_mm[0], ext_mm[1], *max_footprint_mm,
                                                                          " (incl. welded bodies {})".format(welded) if welded else ""))
    if hi[2] * 1e3 > max_height_mm + 1e-6:
        v.append("base height {:.1f} mm exceeds {}{}".format(hi[2] * 1e3, max_height_mm, " (incl. welded bodies {})".format(welded) if welded else ""))
    if lo[2] * 1e3 < -floor_tol_mm:
        v.append("base extends {:.2f} mm below the mounting plane".format(-lo[2] * 1e3))
    return _result(v, footprint_mm=[float(ext_mm[0]), float(ext_mm[1])], height_mm=float(hi[2] * 1e3), bottom_mm=float(lo[2] * 1e3), geoms=n, welded_bodies=welded)


# ============================================================================= C4
def c4_behavior_wire(facts, min_probe_pass=0.5, actor_dir=None, audit_is_violation=False):
    """C4: the actor was driven over the wire in the grader's physics and only moved the robot.

    `facts` is what the behavior grader recorded while running the actor (it, not this
    function, executes agent code): {physics_owner: "grader", actor_spawned_unprivileged: bool,
    actuators_written: [names], robot_actuators: [names], success_source: "grader_predicate",
    env_success_consulted: bool, probes_total: int, probes_passed: int, episodes_clean: int}.
    Optional `actor_dir` adds a static audit (calls to `.success(` / `.set_state(` reported);
    with `audit_is_violation=True` (the behavior grader, CHAIN_PLAN 4.7 anti-cheat "the actor
    cannot call env.success / set_state") any such call fails the check.
    """
    v = []
    if facts.get("physics_owner") != "grader":
        v.append("physics not owned by the grader ({})".format(facts.get("physics_owner")))
    if not facts.get("actor_spawned_unprivileged", False):
        v.append("actor was not spawned as the unprivileged user")
    extra = sorted(set(facts.get("actuators_written", [])) - set(facts.get("robot_actuators", [])))
    if extra:
        v.append("actions reached non-robot actuators {}".format(extra))
    if facts.get("success_source") != "grader_predicate" or facts.get("env_success_consulted", False):
        v.append("success was not recomputed by the grader's own predicate")
    total = int(facts.get("probes_total", 0))
    passed = int(facts.get("probes_passed", 0))
    frac = (passed / total) if total else 0.0
    if total == 0:
        v.append("no probes were issued")
    elif frac < min_probe_pass:
        v.append("probe pass fraction {:.2f} below {}".format(frac, min_probe_pass))
    if int(facts.get("episodes_clean", 1)) < 1:
        v.append("no clean episode")
    audit = {"success_calls": [], "set_state_calls": []}
    if actor_dir and os.path.isdir(actor_dir):
        for dp, _dn, fn in os.walk(actor_dir):
            for f in fn:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dp, f)
                try:
                    tree = ast.parse(open(path).read(), filename=f)
                except (SyntaxError, UnicodeDecodeError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr == "success":
                            audit["success_calls"].append("{}:{}".format(f, node.lineno))
                        if node.func.attr == "set_state":
                            audit["set_state_calls"].append("{}:{}".format(f, node.lineno))
    if audit_is_violation and (audit["success_calls"] or audit["set_state_calls"]):
        v.append("actor source reaches for a simulator's verdict / state: success {} set_state {}".format(
            audit["success_calls"][:4], audit["set_state_calls"][:4]))
    return _result(v, probe_pass_fraction=frac, probes_total=total, audit=audit)
