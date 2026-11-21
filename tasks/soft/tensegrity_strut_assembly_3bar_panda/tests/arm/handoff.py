"""Hand-off adapters A1..A6 of the arm exemplar chain (CHAIN_PLAN.md sections 2.3-2.4).

Every adapter is a pure function of its declared inputs: no hidden state, no clock, no
network, deterministic output (byte-identical files for identical inputs, which is what the
CI rule `adapter(oracle_k) == given_{k+1}` of section 2.3 asserts). The only side effects are
the declared output files under `out_dir`. Every emitted JSON carries `handoff_version`.

Units (section 2.1): hardware-design and assembly files are mm / rad / kg; robot-description
and later are metres / rad; the adapters convert (`MM` = 1e-3).

  A1 requirements_to_hardware_design      requirements.yaml, bom.json, kinematic_sketch.json, design_rules.yaml (mm)
     a1_requirements_to_cad                 the same with the stand-in variants block as the map
  A2 hardware_design_to_assembly          parts_manifest.json, mating_graph.json, meshes/
  A3 hardware_design_to_robot_description assembly.json (SI), meshes/, hulls/, actuator_facts.json, gripper.json
  A4 robot_description_to_control         interface.yaml, ik_interface.py, practice_targets.json, model.mjb
  A5 control_and_description_to_environment  env_givens/ (canonical robot subtree, ik, action space)
  A6 environment_to_behavior              wire.md, obs_action_schema.json, env package without oracle files
"""

import copy
import json
import os
import shutil

import numpy as np

from . import HANDOFF_VERSION
from .chain_model import JOINT_ORDER, Chain, driven_by_list, servo_joint_index
from .frames import (apply, frame_with_z, inv, matrix_to_rpy, pose, pose_from_dict, pose_to_dict,
                     quat_from_matrix, round_list)
from .geometry import (PLA_DENSITY_KG_M3, box_inertia, convex_pieces, export_stl, load_step_solid,
                       mesh_mass_properties, servo_collision_primitives, sha256_file, shift_inertia,
                       tessellate)
from .gripper import kit_in_carrier, load_kit

__all__ = [
    "HANDOFF_VERSION", "MM", "ARM_JOINTS", "JAW_JOINT", "TOOL_POINT_SERVO_MM", "TOOL_APPROACH_SERVO",
    "requirements_to_hardware_design", "a1_requirements_to_cad", "hardware_design_to_assembly",
    "hardware_design_to_robot_description", "robot_description_to_control",
    "control_and_description_to_environment", "environment_to_behavior",
    "actuator_facts_from_bom", "load_json", "dump_json", "tool_from_kit", "design_digest",
    "MATE_TOL_POS_MM", "MATE_TOL_ANG_DEG", "MATE_WEIGHTS",
]

MM = 1e-3
ARM_JOINTS = list(JOINT_ORDER)
JAW_JOINT = "jaw"
# requirements.yaml gripper.tool_frame: tool point 25 mm beyond the horn centre along +y of S6
TOOL_POINT_SERVO_MM = [0.0, 32.0, 46.0]
TOOL_APPROACH_SERVO = [0.0, 1.0, 0.0]
MATE_TOL_POS_MM, MATE_TOL_ANG_DEG = 1.5, 3.0
MATE_WEIGHTS = {"sleeve_seat": 1.0, "horn_mate": 1.5, "part_seat": 1.0, "base_mount": 1.0}
LATCH = {"consecutive_ticks": 50, "tick_s": 0.01, "relative_speed_max_mm_s": 5.0,
         "note": "a mate is achieved when its predicate holds for 50 consecutive 10 ms ticks with relative speed < 5 mm/s; the grader then welds at the DESIGNED relative pose and records the pre-snap error (CHAIN_PLAN 2.5, 05 B6)"}


# ----------------------------------------------------------------------------- small helpers
def load_json(path_or_obj):
    if isinstance(path_or_obj, (dict, list)):
        return path_or_obj
    with open(path_or_obj) as f:
        return json.load(f)


def dump_json(obj, path):
    """Deterministic JSON: fixed key order as built, 2-space indent, trailing newline."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
    return path


def tool_from_kit(kit=None):
    """The tool convention as `Chain` wants it (servo id, point and approach in the S6 frame)."""
    if kit is None:
        return {"servo_id": "S6", "point": list(TOOL_POINT_SERVO_MM), "approach": list(TOOL_APPROACH_SERVO)}
    T = pose_from_dict(kit["tcp"]["in_servo"])
    return {"servo_id": kit["joint"]["driven_by"], "point": round_list(T[:3, 3], 9),
            "approach": round_list(T[:3, 2], 9)}


def actuator_facts_from_bom(bom):
    """Per-occurrence actuator facts from a CAD-facing bom.json (pre- or post-realignment spelling).

    Returns {occurrence id: {catalogue_id, standin_variant, role, mass_kg, stall_torque_nm,
    no_load_speed_rad_s, voltage_v}}. The values are the catalogue's copies (C0 verifies).
    """
    bom = load_json(bom)
    out = {}
    for a in bom.get("actuators", []):
        stall = a.get("stall_torque_nm", a.get("stall_torque_Nm"))
        out[a["id"]] = {
            "catalogue_id": a.get("catalogue_id", a.get("part")),
            "standin_variant": a.get("standin_variant"),
            "role": a.get("role"),
            "mass_kg": float(a["mass_kg"]),
            "stall_torque_nm": float(stall) if stall is not None else None,
            "no_load_speed_rad_s": (float(a["no_load_speed_rad_s"]) if a.get("no_load_speed_rad_s") is not None else None),
            "voltage_v": float(a.get("voltage_V", a.get("voltage_v", 12.0))),
        }
    return out


def design_digest(parts_dir, assembly_json_path):
    """sha256 per file of a hardware-design output (for C-A and the manifests)."""
    files = {}
    for name in sorted(os.listdir(parts_dir)):
        p = os.path.join(parts_dir, name)
        if os.path.isfile(p):
            files["parts/" + name] = sha256_file(p)
    files["assembly.json"] = sha256_file(assembly_json_path)
    return files


def _resolve_servo(servo_standin):
    """`servo_standin` is a dir with servo_standin.{json,step} or a {json, step} dict."""
    if isinstance(servo_standin, dict) and "json" in servo_standin:
        return servo_standin["json"], servo_standin["step"]
    d = servo_standin
    return os.path.join(d, "servo_standin.json"), os.path.join(d, "servo_standin.step")


def _resolve_kit(gripper_kit):
    return gripper_kit if isinstance(gripper_kit, dict) else load_kit(gripper_kit)


def _link_solids(parts_dir, chain):
    """{link: (solid, mesh)} for every link of the chain (STEP loaded once)."""
    out = {}
    for name in chain.order:
        link = chain.links[name]
        path = os.path.join(parts_dir, os.path.basename(link.part_file))
        solid = load_step_solid(path)
        out[name] = (solid, tessellate(solid))
    return out


def _link_mass_properties(link, mesh, facts, servo_meta, density):
    """Mass (kg), CoM (mm) and inertia about the CoM (kg mm^2) of a link with its servos.

    Reproduces `arm_design_check._inertia_of_link` (:878): printed solid x density (or the
    larger mass_override), each servo as a box of the stand-in size with the OCCURRENCE's
    catalogue mass at the stand-in's centre of mass.
    """
    m_link, c_link, I_link = mesh_mass_properties(mesh, density, link.mass_override)
    parts = [(m_link, c_link, I_link)]
    size = np.asarray(servo_meta["body"]["size"], float)
    com_s = np.asarray(servo_meta.get("centre_of_mass", [0.0, 0.0, size[2] / 2]), float)
    for sid, Tocc in link.occurrences:
        m_s = float(facts[sid]["mass_kg"])
        R = Tocc[:3, :3]
        parts.append((m_s, apply(Tocc, com_s), R @ box_inertia(m_s, size) @ R.T))
    m = sum(p[0] for p in parts)
    c = sum(p[0] * p[1] for p in parts) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in parts:
        I += shift_inertia(Ii, mi, ci - c)
    return m, c, I, m_link


def _mass_block(m, c, I, scale):
    if abs(scale - MM) < 1e-15:
        return {"mass_kg": round(float(m), 12), "com_m": round_list(np.asarray(c) * scale, 12),
                "inertia_kg_m2": [round_list(r, 15) for r in np.asarray(I) * scale * scale]}
    return {"mass_kg": round(float(m), 12), "com_mm": round_list(c, 9),
            "inertia_kg_mm2": [round_list(r, 9) for r in np.asarray(I)]}


# ============================================================================= A1
# Planar sizing kinematics reproduced from the requirements task's public `sizing_model.py`
# (`envelope_grid_poses`, `ik_planar`, `bound_corners`) and `reference_design.corner_poses`, so
# the adapter is a pure function of its inputs and imports nothing from a task folder.
A1_TEMPLATE_BOX_CHAIN_RAD = {
    "base_yaw": (-1.7453, 1.7453), "shoulder_pitch": (-1.8326, 0.7854), "elbow_pitch": (-2.9671, -0.2618),
    "wrist_pitch": (-2.0944, 1.8326), "wrist_roll": (-1.6581, 1.6581),
}
A1_PAD_RAD = 0.0873                      # 5 deg, both ends, before clipping to the template box
A1_SHOULDER_HEIGHT_BAND_M = 0.05         # shoulder axis may sit up to this above the sketch base height
A1_BASE_FOOTPRINT_MAX_MM = [110.0, 110.0]  # reference-kitchen mount interface (C3')
A1_BASE_HEIGHT_MAX_MM = 60.0
A1_BRIEF_DEFAULTS = {                    # the arb_arm_a1 brief (oracle/brief_params.json); overridden by brief_params
    "payload_kg": {"min": 0.20, "nominal": 0.30, "max": 0.35},
    "tool_pitch_rad": [-1.5708, -0.7854],
    "torque_utilisation_max": 0.70,
}
A1_TOLERANCE = {"position_mm": 2.0, "approach_axis_deg": 3.0}
A1_SKETCH_JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "gripper"]
A1_AXIS_AT_ZERO = {"base_yaw": [0, 0, 1], "shoulder_pitch": [1, 0, 0], "elbow_pitch": [1, 0, 0],
                   "wrist_pitch": [1, 0, 0], "wrist_roll": [0, 0, 1]}
A1_ASSEMBLY_ABSTRACTION = {"shrink_body_mm": 1.0, "shrink_ears_mm": 0.3, "shrink_cable_block_mm": 0.5,
                           "sweep_mm": 48.0, "step_mm": 2.0, "volume_threshold": 1.0e-3}   # sweep = body height 46 + 2


def _a1_ik_planar(r, z, pitch, lengths, base_height):
    """Elbow-up planar IK of the requirements template (sizing_model.ik_planar), template angles."""
    import math
    l1, l2, l3 = lengths
    wr, wz = r - l3 * math.cos(pitch), z - l3 * math.sin(pitch)
    vr, vz = wr, wz - base_height
    dist = math.hypot(vr, vz)
    if dist > l1 + l2 + 1e-12 or dist < abs(l1 - l2) - 1e-12 or dist < 1e-9:
        return None
    cos_elbow = max(-1.0, min(1.0, (l1 * l1 + l2 * l2 - dist * dist) / (2.0 * l1 * l2)))
    q3 = -(math.pi - math.acos(cos_elbow))
    cos_alpha = max(-1.0, min(1.0, (l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)))
    q2 = math.atan2(vz, vr) + math.acos(cos_alpha)
    return q2, q3, pitch - (q2 + q3)


def _a1_envelope_poses(env):
    """(r, z, pitch) of sizing_model.envelope_grid_poses(env, 9, 9, 3) + reference_design.corner_poses."""
    import math
    poses = []
    rs = np.linspace(env["radius_m"][0], env["radius_m"][1], 9)
    zs = np.linspace(env["height_m"][0], env["height_m"][1], 9)
    for r in rs:
        for z in zs:
            for _yaw in range(3):
                for pitch in env["tool_pitch_rad"]:
                    for _roll in range(2):
                        poses.append((float(r), float(z), float(pitch)))
    r0, r1 = env["radius_m"]
    z0, z1 = env["height_m"]
    p_down, p_tilt = env["tool_pitch_rad"][0], env["tool_pitch_rad"][1]
    for r in (r0, r1):
        for z in (z0, z1):
            for _k in range(2):
                poses.append((r, z, p_down))
    poses += [(r1, 0.5 * (z0 + z1), p_tilt), (r1, 0.5 * (z0 + z1), p_tilt), (r1, z0, p_tilt), (r1, z0, p_tilt)]
    return [(round(r, 6), round(z, 6), round(p, 6)) for r, z, p in poses]


def _a1_requirement_lookup(requirements_yaml):
    """{quantity: (comparator, bound_SI)} of the stage-1 requirements (units converted to SI)."""
    import math
    factors = {"kg": 1.0, "g": 1e-3, "m": 1.0, "mm": 1e-3, "cm": 1e-2, "rad": 1.0, "deg": math.pi / 180.0,
               "s": 1.0, "ms": 1e-3, "min": 60.0, "USD": 1.0, "V": 1.0, "N": 1.0, "N.m": 1.0, "Nm": 1.0,
               "N*m": 1.0, "N-m": 1.0, "kg.cm": 0.0980665, "kg*cm": 0.0980665, "rad/s": 1.0,
               "rpm": 2.0 * math.pi / 60.0, "deg/s": math.pi / 180.0, "count": 1.0, "": 1.0, "1": 1.0}
    out = {}
    for it in requirements_yaml.get("requirements", []):
        f = factors.get(str(it.get("unit", "")), 1.0)
        out.setdefault(it["quantity"], []).append((it["comparator"], float(it["bound"]) * f, it))
    return out


def _a1_yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = repr(v)
        return s
    if isinstance(v, int):
        return str(v)
    s = str(v)
    if s == "" or any(ch in s for ch in ":#{}[],&*!|>'\"%@`") or s.strip() != s or s.lower() in ("yes", "no", "true", "false", "null", "~"):
        return json.dumps(s)
    return s


def _a1_yaml_list_inline(values):
    return "[" + ", ".join(_a1_yaml_scalar(v) for v in values) + "]"


def _a1_block(text, indent):
    """A folded block scalar (`>`) with the given indent."""
    pad = " " * indent
    lines = [pad + ln.rstrip() for ln in text.strip().splitlines()]
    return ">\n" + "\n".join(lines) + "\n"


def _a1_range_used(sketch, env, heights):
    """joint_range_used_rad in the chain convention, padded and clipped to the template box."""
    import math
    lengths = []
    for a in ("min", "max"):
        for b in ("min", "max"):
            for c in ("min", "max"):
                lengths.append((float(sketch["links"]["link_1_m"][a]), float(sketch["links"]["link_2_m"][b]),
                                float(sketch["links"]["link_3_m"][c])))
    poses = _a1_envelope_poses(env)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n_miss = 0
    for h in heights:
        for L in lengths:
            for r, z, pitch in poses:
                q = _a1_ik_planar(r, z, pitch, L, h)
                if q is None:
                    n_miss += 1
                    continue
                qc = np.array([q[0] - math.pi / 2.0, q[1], q[2]])
                lo = np.minimum(lo, qc)
                hi = np.maximum(hi, qc)
    if not np.all(np.isfinite(lo)):
        raise ValueError("A1: no envelope pose is reachable by the sketch (planar IK)")
    yaw_half = 0.5 * (env["yaw_rad"][1] - env["yaw_rad"][0])
    roll_half = 0.5 * (env["tool_roll_rad"][1] - env["tool_roll_rad"][0])
    raw = {"base_yaw": (-yaw_half, yaw_half), "shoulder_pitch": (lo[0], hi[0]), "elbow_pitch": (lo[1], hi[1]),
           "wrist_pitch": (lo[2], hi[2]), "wrist_roll": (-roll_half, roll_half)}
    out = {}
    for name, (a, b) in raw.items():
        box = A1_TEMPLATE_BOX_CHAIN_RAD[name]
        a = max(box[0], math.floor((a - A1_PAD_RAD) * 1e4) / 1e4)
        b = min(box[1], math.ceil((b + A1_PAD_RAD) * 1e4) / 1e4)
        out[name] = [float(a), float(b)]
    return out, {"poses": len(poses), "link_corners": len(lengths), "heights_m": list(heights),
                 "unreachable_evaluations": n_miss, "raw_rad": dict((k, [float(v[0]), float(v[1])]) for k, v in raw.items())}


def requirements_to_hardware_design(brief_params, requirements_yaml, bom_json, kinematic_sketch,
                                    catalogue, standin_map, out_dir=None, servo_standin_json=None):
    """A1: the requirements stage's SI output -> the CAD task's mm givens (CHAIN_PLAN 2.3 / 3.2).

    Implemented by the CAD realignment agent (build order step 2). Inputs (paths or parsed
    objects; `requirements_yaml` may also be YAML text):
      brief_params        oracle/brief_params.json of arm_requirements_bom_from_brief (SI); None -> the
                          arb_arm_a1 brief constants in A1_BRIEF_DEFAULTS (payload band bottom / nominal,
                          the two grasp pitches): everything else comes from the three stage-1 files
      requirements_yaml   stage-1 requirements.yaml: requirements[] {id REQ-*, quantity, comparator, bound, unit, source_clause}
      bom_json            stage-1 bom.json: {currency, bus_voltage_v, parts[] {part_id, count, role}, total_cost_usd}
      kinematic_sketch    stage-1 kinematic_sketch.json: {dof, base_height_m, links{link_k_m{min,max}}, joints{name{servo_id,count}}}
      catalogue           component_catalogue.json (the single source of mass / stall / speed / voltage / price)
      standin_map         standin_map.json: {catalogue_id: standin_variant} e.g. FT-STS3215-12V -> SVX-3215, FT-STS3046 -> SVX-3046;
                          or a servo_standin.json whose `variants` block names the catalogue id per variant
      servo_standin_json  optional servo_standin.json (parsed or path) whose `variants` block is checked against the map
    Returns a dict with keys "requirements.yaml", "bom.json", "kinematic_sketch.json",
    "design_rules_patch" and "design_rules.yaml" (parsed objects, mm / rad / kg / N m), plus
    "texts" {filename: exact file text} and "range_report"; writes the four files under out_dir
    when given (byte-identical for identical inputs: the CI rule of section 2.3).
    Rules (chain_contract.json adapters[A1].rules; CHAIN_PLAN 3.2):
      * joint_range_used_rad = min / max of the planar IK solutions (sizing_model.ik_planar) over
        sizing_model.envelope_grid_poses(envelope, 9, 9, 3) x both tool pitches x both rolls plus
        the 12 corner poses, evaluated at all 8 link-bound corners (sizing_model.bound_corners) and
        both ends of the shoulder-height band, mapped to the chain convention
        (q_shoulder = theta - pi/2, others identity), padded +-0.0873 rad, clipped to the template
        box {base_yaw +-1.7453, shoulder [-1.8326, 0.7854], elbow [-2.9671, -0.2618],
        wrist [-2.0944, 1.8326], wrist_roll +-1.6581}; yaw and roll evaluate to +-1.6581;
      * occurrence ids: count 1 -> S<k>; count 2 -> S<k>A, S<k>B (chain_model.occurrence_ids);
      * every actuator fact is COPIED from the catalogue through standin_map (never re-derived);
      * dimensions in mm: shoulder axis height [h0*1000, h0*1000 + 50], link bounds x1000,
        wrist_pitch_axis_to_tcp band, base footprint [110, 110], base height 60 unchanged;
      * reach_envelope from the requirements (r, z, yaw, roll) and the brief's grasp pitches:
        r [120, 300], z [20, 180], azimuth [-90, 90] about +y, elevation {-90, -45} discrete,
        roll free, tolerance 2 mm / 3 deg;
      * payload nominal 0.30, band [0.20, 0.35], utilisation 0.70; moving_mass_budget 0.80;
      * project = "arb_arm_a1"; requirement ids REQ-01..REQ-16 copied.
    Deterministic; `handoff_version` in every emitted object.
    """
    import math
    from .chain_model import occurrence_ids
    # ---- inputs ---------------------------------------------------------------------------
    if isinstance(requirements_yaml, str):
        import yaml
        if os.path.isfile(requirements_yaml):
            with open(requirements_yaml) as f:
                requirements_yaml = yaml.safe_load(f)
        else:
            requirements_yaml = yaml.safe_load(requirements_yaml)
    bom1 = load_json(bom_json)
    sketch = load_json(kinematic_sketch)
    cat = load_json(catalogue)
    brief = dict(A1_BRIEF_DEFAULTS)
    if brief_params:
        bp = load_json(brief_params)
        if "payload_kg" in bp:
            brief["payload_kg"] = dict(bp["payload_kg"])
        if "envelope" in bp and "tool_pitch_rad" in bp["envelope"]:
            brief["tool_pitch_rad"] = list(bp["envelope"]["tool_pitch_rad"])
    smap = load_json(standin_map) if standin_map is not None else {}
    if "variants" in smap:                      # a servo_standin.json was passed instead of the map
        servo_standin_json = servo_standin_json or smap
        smap = dict((v["catalogue_id"], name) for name, v in smap["variants"].items())
    variants = None
    if servo_standin_json is not None:
        variants = load_json(servo_standin_json).get("variants", {})
    cat_by_id = {}
    for group in ("servos", "bus_boards", "psus"):
        for item in cat.get(group, []):
            cat_by_id[item["id"]] = item
    req = _a1_requirement_lookup(requirements_yaml)

    def bound(quantity, default=None):
        rows = req.get(quantity)
        if not rows:
            if default is None:
                raise ValueError("A1: requirements.yaml has no `{}` entry".format(quantity))
            return default
        return rows[0][1]

    # ---- the envelope in SI (requirements) + the brief's discrete grasp pitches -------------
    r_min, r_max = bound("reach_radius_min"), bound("reach_radius_max")
    z_min, z_max = bound("workspace_height_min"), bound("workspace_height_max")
    yaw_half = bound("yaw_half_range")
    roll_half = bound("tool_roll_half_range")
    pitches = sorted(float(p) for p in (
        [row[1] for row in req["grasp_pitch_angles"]] if req.get("grasp_pitch_angles") else brief["tool_pitch_rad"]))
    env = {"radius_m": [r_min, r_max], "height_m": [z_min, z_max], "yaw_rad": [-yaw_half, yaw_half],
           "tool_pitch_rad": pitches, "tool_roll_rad": [-roll_half, roll_half]}
    h0 = float(sketch["base_height_m"])
    ranges, range_report = _a1_range_used(sketch, env, (h0, h0 + A1_SHOULDER_HEIGHT_BAND_M))
    payload_max = bound("payload_mass")
    moving_budget = bound("moving_mass", 0.8)
    bus_v = float(bom1.get("bus_voltage_v", bound("bus_voltage", 12.0)))
    print_len_m = bound("max_printed_length", 0.22)

    # ---- occurrences and catalogue facts ----------------------------------------------------
    joints_sk = sketch["joints"]
    occ = []            # (id, joint, catalogue_id)
    driven_by = {}
    for k, jname in enumerate(A1_SKETCH_JOINTS, start=1):
        entry = joints_sk[jname]
        count = int(entry.get("count", 1))
        ids = occurrence_ids(k, count)
        driven_by[jname] = ids
        for sid in ids:
            occ.append((sid, jname, entry["servo_id"]))

    def facts_of(cid):
        c = cat_by_id.get(cid)
        if c is None or c.get("kind") != "servo":
            raise ValueError("A1: sketch servo {} is not a catalogue servo".format(cid))
        if cid not in smap:
            raise ValueError("A1: no stand-in variant for catalogue servo {} (standin_map)".format(cid))
        if variants is not None:
            vv = variants.get(smap[cid])
            if vv is None or vv.get("catalogue_id") != cid:
                raise ValueError("A1: servo_standin.json variants block does not name {} for {}".format(cid, smap[cid]))
        return c

    stall_per_joint = {}
    actuators = []
    role_of = {"base_yaw": "base_yaw actuator", "shoulder_pitch": "shoulder_pitch actuator",
               "elbow_pitch": "elbow_pitch actuator", "wrist_pitch": "wrist_pitch actuator",
               "wrist_roll": "wrist_roll actuator", "gripper": "gripper actuator (carried, no arm joint)"}
    for sid, jname, cid in occ:
        c = facts_of(cid)
        n = len(driven_by[jname])
        role = role_of[jname] + (" (dual: {} of 2, coaxial with the other, horns facing each other)".format("A" if sid.endswith("A") else "B") if n == 2 else "")
        actuators.append({
            "id": sid, "catalogue_id": cid, "standin_variant": smap[cid], "joint": jname, "role": role,
            "cad": "servo_standin.step", "mass_kg": float(c["mass_kg"]), "stall_torque_nm": float(c["stall_torque_nm"]),
            "no_load_speed_rad_s": float(c["no_load_speed_rad_s"]), "voltage_range_v": [float(v) for v in c["voltage_range_v"]],
            "voltage_v": bus_v, "price_usd": float(c["price_usd"]), "protocol": c.get("protocol"),
        })
        stall_per_joint[jname] = round(stall_per_joint.get(jname, 0.0) + float(c["stall_torque_nm"]), 6)
    electronics = []
    for line in bom1.get("parts", []):
        if line.get("role") in ("bus_board", "psu"):
            c = cat_by_id.get(line["part_id"], {})
            electronics.append({"id": line["part_id"], "role": line["role"], "count": int(line["count"]),
                                "price_usd": float(c.get("price_usd", 0.0)), "note": "not used by the CAD grader; kept for traceability (C0)"})
    n_horns = sum(len(driven_by[j]) for j in ARM_JOINTS)
    n_servos = len(occ)
    total_cost = float(bom1.get("total_cost_usd", sum(a["price_usd"] for a in actuators) + sum(e["price_usd"] * e["count"] for e in electronics)))

    # ---- mm dimensions ----------------------------------------------------------------------
    L = sketch["links"]
    mm = lambda v: round(float(v) * 1000.0, 3)
    dims = {
        "shoulder_pitch_axis_height_mm": [mm(h0), mm(h0 + A1_SHOULDER_HEIGHT_BAND_M)],
        "upper_arm_axis_distance_mm": [mm(L["link_1_m"]["min"]), mm(L["link_1_m"]["max"])],
        "forearm_axis_distance_mm": [mm(L["link_2_m"]["min"]), mm(L["link_2_m"]["max"])],
        "wrist_pitch_axis_to_tcp_mm": [mm(L["link_3_m"]["min"]), mm(L["link_3_m"]["max"])],
        "base_footprint_max_mm": list(A1_BASE_FOOTPRINT_MAX_MM),
        "base_height_max_mm": A1_BASE_HEIGHT_MAX_MM,
    }
    reach = {
        "radius_from_yaw_axis_mm": [mm(r_min), mm(r_max)], "z_mm": [mm(z_min), mm(z_max)],
        "azimuth_deg": [round(-math.degrees(yaw_half), 3), round(math.degrees(yaw_half), 3)],
        "approach_elevation_deg": [round(math.degrees(p), 3) for p in pitches],
        "roll_deg": [round(-math.degrees(roll_half), 3), round(math.degrees(roll_half), 3)],
        "tolerance": dict(A1_TOLERANCE),
    }
    payload = {"nominal_kg": float(brief["payload_kg"]["nominal"]),
               "static_check_band_kg": [float(brief["payload_kg"]["min"]), float(payload_max)],
               "torque_utilisation_max": float(brief["torque_utilisation_max"])}
    req_ids = [{"id": it["id"], "quantity": it["quantity"], "comparator": it["comparator"], "bound": float(it["bound"]),
                "unit": it["unit"]} for it in requirements_yaml.get("requirements", [])]
    handoff = {"version": HANDOFF_VERSION, "adapter": "A1 requirements_to_hardware_design",
               "source_stage": "requirements", "project": "arb_arm_a1"}

    # ---- parsed objects -------------------------------------------------------------------
    joints_out = []
    for jname in ARM_JOINTS:
        joints_out.append({"name": jname, "type": "revolute", "driven_by": list(driven_by[jname]),
                           "servo_count": len(driven_by[jname]), "required_range_rad": list(ranges[jname]),
                           "axis_at_zero": list(A1_AXIS_AT_ZERO[jname]), "stall_torque_Nm": stall_per_joint[jname]})
    requirements_out = {
        "project": "arb_arm_a1", "stage": "requirements", "handoff": handoff,
        "units": {"length": "mm", "angle": "rad", "mass": "kg", "torque": "N*m"},
        "joints": joints_out,
        "gripper": {"servo_id": driven_by["gripper"][0], "stall_torque_Nm": stall_per_joint["gripper"],
                    "tool_frame": {"point_in_servo_frame": list(TOOL_POINT_SERVO_MM),
                                   "approach_axis_in_servo_frame": list(TOOL_APPROACH_SERVO)}},
        "payload": payload, "moving_mass_budget_kg": float(moving_budget), "dimensions": dims,
        "reach_envelope": reach, "stall_torque_per_joint_Nm": dict((j, stall_per_joint[j]) for j in A1_SKETCH_JOINTS),
        "max_printed_length_mm": mm(print_len_m), "requirement_ids": req_ids,
    }
    bom_out = {
        "stage": "requirements", "project": "arb_arm_a1", "handoff": handoff,
        "description": "Bill of materials fixed by the requirements stage, projected onto the CAD stage: {} servo occurrences of {} catalogue SKUs (facts copied from component_catalogue.json through the stand-in variants), bus electronics, fasteners, PLA. One licence-safe stand-in geometry (servo_standin.step) serves every occurrence; bom.json names the variant, servo_standin.json carries the per-variant facts.".format(
            n_servos, len(set(a["catalogue_id"] for a in actuators))),
        "currency": "USD", "bus_voltage_v": bus_v, "total_cost_usd": total_cost,
        "actuators": actuators, "electronics": electronics,
        "fasteners": [
            {"id": "F-M2x6", "use": "horn to link, 4 per horn ({} horns drive the five arm joints)".format(n_horns), "qty": 4 * n_horns, "clearance_hole_mm": [2.2, 2.8]},
            {"id": "F-M3x8-selftap", "use": "ear to carrier boss, 4 per servo ({} servos)".format(n_servos), "qty": 4 * n_servos, "pilot_hole_mm": [2.5, 2.9]},
            {"id": "F-M4x12", "use": "base to table", "qty": 4, "clearance_hole_mm": [4.3, 4.6]},
        ],
        "material": {"id": "PLA", "density_kg_m3": 1240, "note": "all links and the base are printed PLA"},
        "catalogue_facts_note": "mass_kg, stall_torque_nm and no_load_speed_rad_s are the catalogue's values per occurrence; the grader uses them (per occurrence, summed per joint) for the static torque check; identity check C0 verifies the copy",
    }
    deg = lambda pair: [round(math.degrees(pair[0]), 2), round(math.degrees(pair[1]), 2)]
    chain_list = [
        {"link": "base", "fixed_to": "table", "carries": [driven_by["base_yaw"][0]]},
        {"joint": "base_yaw", "axis_world_at_zero": [0, 0, 1], "driven_by": driven_by["base_yaw"], "required_range_deg": deg(ranges["base_yaw"]), "required_range_rad": ranges["base_yaw"], "note": "horn up; the shoulder bolts to the S1 horn"},
        {"link": "shoulder", "carries": driven_by["shoulder_pitch"], "note": "turns with base_yaw; holds the two shoulder_pitch servos coaxially with a horizontal horn axis (dual_servo_joints)"},
        {"joint": "shoulder_pitch", "axis_world_at_zero": [1, 0, 0], "driven_by": driven_by["shoulder_pitch"], "indicative_height_mm": dims["shoulder_pitch_axis_height_mm"][0], "height_band_mm": dims["shoulder_pitch_axis_height_mm"], "required_range_deg": deg(ranges["shoulder_pitch"]), "required_range_rad": ranges["shoulder_pitch"]},
        {"link": "upper_arm", "carries": driven_by["elbow_pitch"], "indicative_axis_distance_mm": dims["upper_arm_axis_distance_mm"][0], "axis_distance_band_mm": dims["upper_arm_axis_distance_mm"]},
        {"joint": "elbow_pitch", "axis_world_at_zero": [1, 0, 0], "driven_by": driven_by["elbow_pitch"], "required_range_deg": deg(ranges["elbow_pitch"]), "required_range_rad": ranges["elbow_pitch"], "note": "one-sided: the forearm only ever bends forward (negative q); the range comes from the envelope sweep"},
        {"link": "forearm", "carries": driven_by["wrist_pitch"], "indicative_axis_distance_mm": dims["forearm_axis_distance_mm"][0], "axis_distance_band_mm": dims["forearm_axis_distance_mm"]},
        {"joint": "wrist_pitch", "axis_world_at_zero": [1, 0, 0], "driven_by": driven_by["wrist_pitch"], "required_range_deg": deg(ranges["wrist_pitch"]), "required_range_rad": ranges["wrist_pitch"]},
        {"link": "wrist", "carries": driven_by["wrist_roll"], "note": "the roll servo's horn axis is the tool axis (+z at zero); wrist_pitch axis to TCP within the band"},
        {"joint": "wrist_roll", "axis_world_at_zero": [0, 0, 1], "driven_by": driven_by["wrist_roll"], "required_range_deg": deg(ranges["wrist_roll"]), "required_range_rad": ranges["wrist_roll"]},
        {"link": "gripper_carrier", "carries": driven_by["gripper"], "note": "last link; the tool frame follows the {} occurrence (requirements.yaml gripper.tool_frame); wrist_pitch axis to TCP band {} mm".format(driven_by["gripper"][0], dims["wrist_pitch_axis_to_tcp_mm"])},
    ]
    sketch_out = {
        "stage": "requirements", "project": "arb_arm_a1", "handoff": handoff,
        "description": "Kinematic sketch agreed at the requirements stage, projected onto the CAD stage: joint order, axis directions at the zero configuration, which link carries which servo occurrence, the axis-distance bands and the required joint ranges. Positions are indicative; the link design fixes the exact numbers in assembly.json.",
        "units": {"length": "mm", "angle": "rad"},
        "source_sketch_si": {"base_height_m": h0, "links": {k: {"min": float(L[k]["min"]), "max": float(L[k]["max"])} for k in ("link_1_m", "link_2_m", "link_3_m")},
                             "joints": {j: {"servo_id": joints_sk[j]["servo_id"], "count": int(joints_sk[j].get("count", 1)), "occurrences": driven_by[j]} for j in A1_SKETCH_JOINTS}},
        "world_frame": "base link frame, z up, origin on the mounting plane at the base_yaw axis; +y is forward (the azimuth origin of the reach envelope), +x is the direction of the pitch axes at yaw 0",
        "zero_configuration": "upper arm, forearm and wrist vertical (+z); tool approach axis +z; positive q is a right-hand rotation about the joint axis, so reaching forward (+y) is a negative shoulder_pitch",
        "chain": chain_list,
        "dual_servo_joints": "shoulder_pitch and elbow_pitch are each driven by two servos of the same SKU ({} / {}). The two horn axes must be coaxial within 0.5 mm / 1 deg and coincide with the joint axis; the horns face each other with the moving link between them, each horn bolted to its own horn plate of the moving link; both rear bosses are seated in the carrier. The two servos are commanded as one joint and the grader sums their stall torques.".format(
            joints_sk["shoulder_pitch"]["servo_id"], joints_sk["elbow_pitch"]["servo_id"]),
        "servo_placement_freedom": "Any link may carry the servo that drives a joint (parent-side carrier with the child on the horn, or child-side carrier with the parent on the horn). The joint axis must coincide with the horn axis of every driving servo within 1 degree / 0.5 mm.",
        "double_support": "Pitch joints should engage the rear boss of the driving servo with the moving link (double support) where the geometry allows; a joint may be horn-supported with the rear boss seated in the carrier (the reference does this for the wrist, yaw and roll).",
        "sign_conventions": "rpy is fixed-axis XYZ (R = Rz(yaw) Ry(pitch) Rx(roll)); a joint origin is the child frame in the parent frame at q = 0; a joint axis is expressed in the child frame; positive q is a right-hand rotation about the axis; at q = 0 every joint axis equals its axis_world_at_zero (world frame).",
        "joint_range_used_rad": ranges,
        "range_derivation": "planar elbow-up IK (sizing_model.ik_planar) over the envelope grid (9 radii x 9 heights x 3 yaws x 2 pitches x 2 rolls) plus the 12 corner poses, at all 8 link-bound corners and both ends of the shoulder-height band, chain convention, padded +-0.0873 rad, clipped to the template box",
    }
    design_rules_patch = {
        "joint_limits_required": dict((j, list(ranges[j])) for j in ARM_JOINTS),
        "servo_mounting": {"dual": "Dual-servo joints (shoulder_pitch, elbow_pitch): the two servos are coaxial (horn axes within 0.5 mm / 1 deg of each other and of the joint axis) with their horns facing each other; each horn carries its own horn plate of the moving link (boss clearance >= 10.4 mm, four 2.2-2.8 mm holes on the 16 mm PCD, material >= 2 mm around the bolt circle); both rear bosses are seated in the carrier (8.2-8.8 mm holes); both servos are retained by their ears. The keep-in / keep-out probes run per occurrence and the joint's torque allowance is 0.70 x the sum of the two stall torques."},
        "assembly_abstraction": dict(A1_ASSEMBLY_ABSTRACTION),
        "operating_range": {"tcp_inside_reach_envelope": True, "no_mesh_below_mounting_plane": True},
    }
    design_rules_text = _a1_design_rules_text(ranges, reach, A1_ASSEMBLY_ABSTRACTION, handoff)
    import yaml as _yaml
    design_rules_obj = _yaml.safe_load(design_rules_text)
    texts = {
        "requirements.yaml": _a1_requirements_text(requirements_out, joints_out, dims, reach, payload, req_ids, handoff, joints_sk, stall_per_joint),
        "bom.json": json.dumps(bom_out, indent=2, allow_nan=False) + "\n",
        "kinematic_sketch.json": json.dumps(sketch_out, indent=2, allow_nan=False) + "\n",
        "design_rules.yaml": design_rules_text,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, text in texts.items():
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(text)
    return {"requirements.yaml": requirements_out, "bom.json": bom_out, "kinematic_sketch.json": sketch_out,
            "design_rules_patch": design_rules_patch, "design_rules.yaml": design_rules_obj,
            "joint_range_used_rad": ranges, "range_report": range_report, "texts": texts}


def a1_requirements_to_cad(requirements_yaml, bom_json, kinematic_sketch_json, catalogue_json, standin_json,
                           brief_params=None, out_dir=None):
    """The CAD realignment agent's spelling of A1 (task order, 2026-09-26): the stand-in map is the
    `variants` block of servo_standin.json; the brief constants default to the arb_arm_a1 brief."""
    standin = load_json(standin_json)
    smap = dict((v["catalogue_id"], name) for name, v in standin.get("variants", {}).items())
    return requirements_to_hardware_design(brief_params, requirements_yaml, bom_json, kinematic_sketch_json,
                                           catalogue_json, smap, out_dir=out_dir, servo_standin_json=standin)


def _a1_requirements_text(req, joints_out, dims, reach, payload, req_ids, handoff, joints_sk, stall):
    import math
    n_occ = sum(j["servo_count"] for j in joints_out) + 1
    skus = sorted(set(joints_sk[j]["servo_id"] for j in A1_SKETCH_JOINTS))
    L = []
    L.append("# requirements.yaml: reference output of the `requirements` stage, projected onto the CAD stage")
    L.append("# by the chain adapter A1 (shared/arm/handoff.py::requirements_to_hardware_design, {}).".format(handoff["version"]))
    L.append("# 5-DoF desktop arm + parallel gripper, PLA links around {} servo occurrences of {} catalogue SKUs".format(n_occ, len(skus)))
    L.append("# ({}), all represented by one licence-safe stand-in geometry (servo_standin.step).".format(", ".join(skus)))
    L.append("# Units: mm, rad, kg, N*m unless a key says otherwise. Generated file: do not edit by hand.")
    L.append("")
    L.append("project: {}".format(req["project"]))
    L.append("stage: requirements")
    L.append("handoff: {version: %s, adapter: %s, source_stage: requirements}" % (json.dumps(handoff["version"]), json.dumps(handoff["adapter"])))
    L.append("units: {length: mm, angle: rad, mass: kg, torque: N*m}")
    L.append("")
    L.append("world_frame: " + _a1_block("The base link frame is the world frame: z up, origin on the mounting plane (table top) at the base-yaw axis, +y forward (the azimuth origin of the reach envelope), +x the direction of the pitch axes at yaw 0. The base link's STEP part must sit on z >= 0.", 2).rstrip("\n"))
    L.append("")
    L.append("zero_configuration: " + _a1_block("With every joint at 0 the upper arm, forearm and wrist point straight up (+z) and the tool approach axis points +z as well; every joint axis then equals its axis_at_zero below. Positive q is a right-hand rotation about the axis, so reaching forward (+y) is a negative shoulder_pitch. The zero pose is a reference pose only (the elbow's required range does not even contain 0); the grader samples the operating range.", 2).rstrip("\n"))
    L.append("")
    L.append("joints:  # serial chain, in this exact order; driven_by lists the servo occurrence(s) of each joint")
    hints = {"base_yaw": "vertical", "shoulder_pitch": "horizontal (+x at yaw 0)", "elbow_pitch": "parallel to shoulder_pitch",
             "wrist_pitch": "parallel to elbow_pitch", "wrist_roll": "along the tool (+z at zero)"}
    for j in joints_out:
        L.append("  - {name: %s, type: revolute, driven_by: %s, required_range_rad: %s, axis_at_zero: %s, stall_torque_Nm: %s, axis_hint: %s}" % (
            j["name"], _a1_yaml_list_inline(j["driven_by"]), _a1_yaml_list_inline(j["required_range_rad"]),
            _a1_yaml_list_inline(j["axis_at_zero"]), _a1_yaml_scalar(j["stall_torque_Nm"]), json.dumps(hints[j["name"]])))
    L.append("  # required_range_rad = joint_range_used_rad of adapter A1: planar IK over the reach envelope at every link-bound")
    L.append("  # corner and both ends of the shoulder-height band, padded 5 deg, clipped to the template box (kinematic_sketch.json")
    L.append("  # range_derivation). Declared limits in assembly.json must contain these ranges.")
    L.append("")
    L.append("gripper:")
    L.append("  servo_id: {}".format(req["gripper"]["servo_id"]))
    L.append("  stall_torque_Nm: {}".format(_a1_yaml_scalar(req["gripper"]["stall_torque_Nm"])))
    L.append("  carried_by: the last link of the chain (the gripper carrier)")
    L.append("  note: " + _a1_block("The gripper jaws themselves are a later stage (the bank's gripper kit bolts to the S6 horn). The tool frame is defined on the S6 occurrence so that every design has a well-defined tool centre point.", 4).rstrip("\n"))
    L.append("  tool_frame:")
    L.append("    point_in_servo_frame: {}      # horn axis point (0, 7, 46) + 25 mm along +y of S6 (jaw tips)".format(_a1_yaml_list_inline(req["gripper"]["tool_frame"]["point_in_servo_frame"])))
    L.append("    approach_axis_in_servo_frame: {}".format(_a1_yaml_list_inline(req["gripper"]["tool_frame"]["approach_axis_in_servo_frame"])))
    L.append("")
    L.append("payload:")
    L.append("  nominal_kg: {}".format(_a1_yaml_scalar(payload["nominal_kg"])))
    L.append("  static_check_band_kg: {}   # the grader evaluates the band top AND one hidden draw in the band".format(_a1_yaml_list_inline(payload["static_check_band_kg"])))
    L.append("  torque_utilisation_max: {}         # gravity torque at every joint <= 0.70 x (sum of the joint's stall torques) in any operating pose".format(_a1_yaml_scalar(payload["torque_utilisation_max"])))
    L.append("")
    L.append("moving_mass_budget_kg: {}   # everything that turns with base_yaw: links 1-5 plus every servo except S1 (REQ-12)".format(_a1_yaml_scalar(req["moving_mass_budget_kg"])))
    L.append("")
    L.append("dimensions:   # bands from the requirements sketch (link bounds x 1000); footprint / height = the reference-kitchen mount interface")
    L.append("  shoulder_pitch_axis_height_mm: {}        # above the mounting plane at q = 0 (sketch base height .. +50)".format(_a1_yaml_list_inline(dims["shoulder_pitch_axis_height_mm"])))
    L.append("  upper_arm_axis_distance_mm: {}          # shoulder_pitch axis to elbow_pitch axis (sketch link_1)".format(_a1_yaml_list_inline(dims["upper_arm_axis_distance_mm"])))
    L.append("  forearm_axis_distance_mm: {}            # elbow_pitch axis to wrist_pitch axis (sketch link_2)".format(_a1_yaml_list_inline(dims["forearm_axis_distance_mm"])))
    L.append("  wrist_pitch_axis_to_tcp_mm: {}             # wrist_pitch axis line to the tool centre point at q = 0 (sketch link_3)".format(_a1_yaml_list_inline(dims["wrist_pitch_axis_to_tcp_mm"])))
    L.append("  base_footprint_max_mm: {}".format(_a1_yaml_list_inline(dims["base_footprint_max_mm"])))
    L.append("  base_height_max_mm: {}                          # top of the base link above the mounting plane".format(_a1_yaml_scalar(dims["base_height_max_mm"])))
    L.append("")
    L.append("reach_envelope:   # targets are drawn uniformly here; practice_targets.json shows 12 visible draws")
    L.append("  description: " + _a1_block("Target = tool position + approach axis. Positions in cylindrical coordinates around the yaw axis (azimuth measured from +y, positive towards -x); the approach axis is cos(el) * r_hat + sin(el) * z_hat with r_hat the horizontal unit vector from the yaw axis to the target and el one of the listed elevations (a tool pointing straight down or outward-down at 45 degrees). Roll about the approach axis is free.", 4).rstrip("\n"))
    L.append("  radius_from_yaw_axis_mm: {}".format(_a1_yaml_list_inline(reach["radius_from_yaw_axis_mm"])))
    L.append("  z_mm: {}".format(_a1_yaml_list_inline(reach["z_mm"])))
    L.append("  azimuth_deg: {}".format(_a1_yaml_list_inline(reach["azimuth_deg"])))
    L.append("  approach_elevation_deg: {}      # DISCRETE set: the brief's two grasp pitches (top-down, 45 deg)".format(_a1_yaml_list_inline(reach["approach_elevation_deg"])))
    L.append("  roll_deg: {}                     # informative: wrist_roll must cover it (required_range_rad above)".format(_a1_yaml_list_inline(reach["roll_deg"])))
    L.append("  tolerance: {position_mm: %s, approach_axis_deg: %s}" % (_a1_yaml_scalar(reach["tolerance"]["position_mm"]), _a1_yaml_scalar(reach["tolerance"]["approach_axis_deg"])))
    L.append("  note: " + _a1_block("This is the workspace of the brief (REQ-02..REQ-07). A 5-DoF arm cannot reach every target with every approach, so the grader scores the reachable fraction relative to a reference design built from the same servo occurrences. The operating range of the arm (interference and torque configurations) is the declared joint box restricted to configurations whose tool centre point lies inside this envelope and whose links stay above the mounting plane (design_rules.yaml operating_range).", 4).rstrip("\n"))
    L.append("")
    L.append("stall_torque_per_joint_Nm:   # catalogue stall x servo count; the torque check uses torque_utilisation_max x this")
    for j in A1_SKETCH_JOINTS:
        L.append("  {}: {}".format(j, _a1_yaml_scalar(req["stall_torque_per_joint_Nm"][j])))
    L.append("")
    L.append("max_printed_length_mm: {}   # REQ-16: every part fits the print bed (design_rules.yaml printer_envelope_mm is the binding rule)".format(_a1_yaml_scalar(req["max_printed_length_mm"])))
    L.append("")
    L.append("requirement_ids:   # the requirements stage's list (traceability keys of the chain), SI as written there")
    for it in req_ids:
        L.append("  - {id: %s, quantity: %s, comparator: %s, bound: %s, unit: %s}" % (
            it["id"], it["quantity"], json.dumps(it["comparator"]), _a1_yaml_scalar(it["bound"]), json.dumps(it["unit"])))
    return "\n".join(L) + "\n"


def _a1_design_rules_text(ranges, reach, shrink, handoff):
    L = []
    L.append("# design_rules.yaml: fabrication and interface rules the grader enforces (given input).")
    L.append("# Generated by the chain adapter A1 ({}) from the requirements stage output; the".format(handoff["version"]))
    L.append("# numeric joint limits come from its envelope sweep. Units: mm unless a key says otherwise.")
    L.append("")
    L.append("handoff: {version: %s, adapter: %s}" % (json.dumps(handoff["version"]), json.dumps(handoff["adapter"])))
    L.append("")
    L.append("material:")
    L.append("  name: PLA")
    L.append("  density_kg_m3: 1240")
    L.append("  process: FDM 3D printing, 0.4 mm nozzle")
    L.append("")
    L.append("printer_envelope_mm: [220, 220, 250]   # per part, any orientation")
    L.append("max_parts: 8                            # one STEP solid per link, base and gripper carrier included")
    L.append("min_wall_mm: 1.2                        # measured by inward ray thickness on surface samples")
    L.append("min_through_hole_mm: 2.0                # checked at every interface hole")
    L.append("")
    L.append("servo_mounting:")
    L.append("  keep_out: " + _a1_block("No link material may lie inside the servo envelope (body, horn boss, ears, cable exit block, rear boss). Intersection volume <= 1e-3 of the servo volume per occurrence, both for the link that carries the servo and for the link attached to its horn. Recommended clearance around the body: 0.4 mm.", 4).rstrip("\n"))
    L.append("  ears: " + _a1_block("A servo is retained by its four ears. Screws enter from above (+z of the servo) through the 3.2 mm ear holes into pilot holes of 2.5-2.9 mm in bosses under the ears. Boss material must surround each pilot hole (ring of radius 3.5 mm, 1 mm under the ear underside, inside link material; the hole centre itself must be empty).", 4).rstrip("\n"))
    L.append("  horn: " + _a1_block("The link attached to a horn rests on the servo top face (z = 46 in the servo frame), is >= 2 mm thick around the bolt circle (probed 1 mm above the face at radius 11.5 mm), has a boss clearance hole >= 10.4 mm and four clearance holes of 2.2-2.8 mm on the 16 mm PCD at 45/135/225/315 degrees.", 4).rstrip("\n"))
    L.append("  rear_bearing: " + _a1_block("The rear boss (8 mm x 2.5 mm) must be seated in a hole of 8.2-8.8 mm, either in the link attached to the horn (double-supported joint) or in the carrier (horn-supported joint). Probed by a ring of radius 6.5 mm at mid-boss height.", 4).rstrip("\n"))
    L.append("  cable: the cable exit block on the -y face needs a window; it is part of the keep-out.")
    L.append("  dual: " + _a1_block("Dual-servo joints (shoulder_pitch, elbow_pitch): the two servos are coaxial (horn axes within 0.5 mm / 1 deg of each other and of the joint axis) with their horns facing each other; each horn carries its own horn plate of the moving link (boss clearance >= 10.4 mm, four 2.2-2.8 mm holes on the 16 mm PCD, material >= 2 mm around the bolt circle); both rear bosses are seated in the carrier (8.2-8.8 mm holes); both servos are retained by their ears. The keep-in / keep-out probes run per occurrence and the joint's torque allowance is 0.70 x the sum of the two stall torques.", 4).rstrip("\n"))
    L.append("")
    L.append("operating_range: " + _a1_block("The operating range is the declared joint box restricted to configurations in which (a) no link and no servo lies below the mounting plane (z = 0 in the base frame, the table the arm is bolted to) and (b) the tool centre point lies inside the reach envelope of requirements.yaml (radius, height and azimuth bands; approach and roll free). The grader maps its hidden Halton stream into the declared box and keeps the first configurations that satisfy both (rejection sampling) for the interference and torque checks; the base link itself must sit on z >= 0.", 2).rstrip("\n"))
    L.append("")
    L.append("clearances:")
    L.append("  static_contact: " + _a1_block("touching faces are allowed (0 mm gap) at joints; the grader requires the child link to be within 0.2 mm of the parent link or of a driving servo at q = 0", 4).rstrip("\n"))
    L.append("  interference: " + _a1_block("Over the operating range, non-adjacent links (including the servos they carry) may intersect by at most 1e-3 of the smaller body volume; adjacent links by at most 1e-2. Score degrades log-linearly to 0 at ten times the threshold.", 4).rstrip("\n"))
    L.append("")
    L.append("joint_limits_required:   # declared limits in assembly.json must contain these (adapter A1 joint_range_used_rad)")
    for j in ARM_JOINTS:
        L.append("  {}: {}".format(j, _a1_yaml_list_inline(ranges[j])))
    L.append("")
    L.append("axes_at_zero:   # world-frame joint axis directions at q = 0 (checked within 2 deg; the sign fixes the meaning of the ranges)")
    for j in ARM_JOINTS:
        L.append("  {}: {}".format(j, _a1_yaml_list_inline(A1_AXIS_AT_ZERO[j])))
    L.append("  tool_approach: [0, 0, 1]")
    L.append("")
    L.append("assembly_abstraction:   # the `assemblable` stat (recorded, not scored) and the assembly stage's collision model")
    L.append("  description: " + _a1_block("Bolts and the horn / rear bosses are locators, not collision geometry. The servo body collision solid is shrunk per face by the values below (ears and cable block by their own values); links are never shrunk. `assemblable` sweeps every servo along its horn axis out of its carrier and every moving link off its driving servo(s) (axially for a single servo, else radially away from the axis) in straight lines of one servo body length (46 + 2 mm) in steps of insertion_step_mm, and reports the fraction of the 13 insertions (8 servos + 5 joints) that never intersect the mating part by more than volume_threshold of the smaller body. A cable slot open towards the sleeve mouth is what lets a servo pass this sweep.", 4).rstrip("\n"))
    L.append("  clearance_shrink_mm: {body: %s, ears: %s, cable_block: %s}" % (_a1_yaml_scalar(shrink["shrink_body_mm"]), _a1_yaml_scalar(shrink["shrink_ears_mm"]), _a1_yaml_scalar(shrink["shrink_cable_block_mm"])))
    L.append("  bosses_are_locators: true")
    L.append("  insertion_sweep_mm: {}".format(_a1_yaml_scalar(shrink["sweep_mm"])))
    L.append("  insertion_step_mm: {}".format(_a1_yaml_scalar(shrink["step_mm"])))
    L.append("  volume_threshold: {}   # of the (shrunk) servo volume, per step".format(_a1_yaml_scalar(shrink["volume_threshold"])))
    L.append("")
    L.append("grading_tolerances:")
    L.append("  axis_angle_deg: 1.0     # joint axis vs every driving horn axis (anti-parallel accepted)")
    L.append("  axis_offset_mm: 0.5     # joint axis line vs horn axis line")
    L.append("  contact_mm: 0.2         # parent/child (or driving servo) proximity at the joint")
    L.append("  axis_at_zero_deg: 2.0   # joint axes and tool approach vs axes_at_zero at q = 0")
    return "\n".join(L) + "\n"


# ============================================================================= A2
def hardware_design_to_assembly(parts_dir, assembly_json, bom_json, servo_standin, gripper_kit, out_dir,
                                max_hulls=96, coacd_threshold=0.01, include_base_mount=True,
                                density=PLA_DENSITY_KG_M3):
    """A2: hardware-design output -> the assembly stage's givens (CHAIN_PLAN 2.3-2.5, 4.3).

    Inputs: `parts_dir` (parts/*.step), `assembly_json` (mm), CAD-facing `bom_json`
    (occurrences with catalogue facts), `servo_standin` (dir or {json, step}), `gripper_kit`
    (kit dir or loaded kit). Writes under `out_dir`:
      parts_manifest.json   units mm; parts[] {name, kind printed|servo|kit, mesh_visual,
                            mesh_collision[] (<= max_hulls CoACD hulls; default 96 at threshold 0.01: the
                            assembly cell's collision-model gate fails the reference design at 16 / 0.05,
                            whose hulls fill the S2 / S3 servo pockets by 2.6-4.7 mm; measured 2026-09-26,
                            assembly lane DESIGN.md section 3), mass_kg, com_mm,
                            inertia_kg_mm2, bbox_mm}; servos carry collision_primitives with the
                            published shrink classes instead of hulls
      mating_graph.json     mates[] {id, kind sleeve_seat|horn_mate|part_seat|base_mount, part_a,
                            part_b, frame_a_in_a, frame_b_in_b, insertion_axis_in_a,
                            insertion_depth_mm, tol_pos_mm 1.5, tol_ang_deg 3.0, weight, requires[]}
                            + scopes (rung-1 prefix / full) + latch constants
      meshes/visual/*.stl, meshes/collision/*.stl   mm, binary STL, deterministic bytes
    Returns {"parts_manifest": ..., "mating_graph": ..., "paths": {...}}.
    """
    asm = load_json(assembly_json)
    facts = actuator_facts_from_bom(bom_json)
    servo_json, servo_step = _resolve_servo(servo_standin)
    servo_meta = load_json(servo_json)
    kit = _resolve_kit(gripper_kit)
    chain = Chain(asm, tool_from_kit(kit))
    vis_dir = os.path.join(out_dir, "meshes", "visual")
    col_dir = os.path.join(out_dir, "meshes", "collision")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(col_dir, exist_ok=True)

    manifest_parts = []
    # printed links
    solids = _link_solids(parts_dir, chain)
    for name in chain.order:
        link = chain.links[name]
        mesh = solids[name][1]
        m, c, I = mesh_mass_properties(mesh, density, link.mass_override)
        vis = export_stl(mesh, os.path.join(vis_dir, name + ".stl"))
        hulls = []
        for i, piece in enumerate(convex_pieces(mesh, max_hulls, coacd_threshold)):
            hulls.append(os.path.relpath(export_stl(piece, os.path.join(col_dir, "{}_{:02d}.stl".format(name, i))), out_dir))
        manifest_parts.append({
            "name": name, "kind": "printed", "part_file": link.part_file,
            "mesh_visual": os.path.relpath(vis, out_dir), "mesh_collision": hulls,
            "mass_kg": round(m, 9), "com_mm": round_list(c, 6), "inertia_kg_mm2": [round_list(r, 9) for r in I],
            "bbox_mm": [round_list(mesh.bounds[0], 6), round_list(mesh.bounds[1], 6)],
            "volume_mm3": round(float(mesh.volume), 6), "material": {"id": "PLA", "density_kg_m3": density},
            "servo_occurrences": [sid for sid, _ in link.occurrences],
        })
    # servo occurrences (one stand-in geometry, per-occurrence catalogue facts)
    servo_mesh = tessellate(load_step_solid(servo_step))
    servo_vis = os.path.relpath(export_stl(servo_mesh, os.path.join(vis_dir, "servo_standin.stl")), out_dir)
    size = np.asarray(servo_meta["body"]["size"], float)
    com_s = np.asarray(servo_meta.get("centre_of_mass", [0.0, 0.0, size[2] / 2]), float)
    prims = servo_collision_primitives(servo_meta)
    for sid, carrier, Tocc in chain.all_occurrences():
        if sid not in facts:
            raise ValueError("bom.json has no actuator occurrence {}".format(sid))
        f = facts[sid]
        manifest_parts.append({
            "name": sid, "kind": "servo", "catalogue_id": f["catalogue_id"], "standin_variant": f["standin_variant"],
            "carrier_link": carrier, "designed_pose_in_carrier": pose_to_dict(Tocc, 1.0, 6, 9),
            "mesh_visual": servo_vis, "mesh_collision": [], "collision_primitives": prims,
            "locators_visual_only": ["horn_boss", "rear_bearing_boss"],
            "mass_kg": f["mass_kg"], "com_mm": round_list(com_s, 6),
            "inertia_kg_mm2": [round_list(r, 9) for r in box_inertia(f["mass_kg"], size)],
            "bbox_mm": [list(servo_meta["body"]["min"]), list(servo_meta["body"]["max"])],
            "stall_torque_nm": f["stall_torque_nm"],
        })
    # gripper kit parts
    kit_dir = kit["_dir"]
    for part in kit["parts"]:
        mesh = tessellate(load_step_solid(os.path.join(kit_dir, part["part_file"])))
        vis = export_stl(mesh, os.path.join(vis_dir, part["name"] + ".stl"))
        hulls = []
        for i, piece in enumerate(convex_pieces(mesh, max_hulls, coacd_threshold)):
            hulls.append(os.path.relpath(export_stl(piece, os.path.join(col_dir, "{}_{:02d}.stl".format(part["name"], i))), out_dir))
        manifest_parts.append({
            "name": part["name"], "kind": "kit", "part_file": "gripper_kit/" + part["part_file"],
            "mesh_visual": os.path.relpath(vis, out_dir), "mesh_collision": hulls,
            "mass_kg": part["mass_kg"], "com_mm": part["com_mm"], "inertia_kg_mm2": part["inertia_kg_mm2"],
            "bbox_mm": part["bbox_mm"], "volume_mm3": part.get("volume_mm3"),
            "material": {"id": "PLA", "density_kg_m3": density},
            "pad": part["pad"],
        })

    assembly_path = assembly_json if isinstance(assembly_json, str) else None
    manifest = {
        "handoff_version": HANDOFF_VERSION, "adapter": "A2 hardware_design_to_assembly",
        "units": {"length": "mm", "angle": "rad", "mass": "kg"},
        "source": {"design_digest": design_digest(parts_dir, assembly_path) if assembly_path else None,
                   "servo_standin_step_sha256": sha256_file(servo_step), "gripper_kit": kit["kit_id"]},
        "tessellation": {"linear_tolerance_mm": 0.05, "angular_tolerance_rad": 0.3, "same_as": "arm_design_check.py TESS_TOL / TESS_ANG"},
        "collision": {"method": "coacd", "max_pieces_per_part": max_hulls, "threshold": coacd_threshold,
                      "servo_shrink_mm": {"body": 1.0, "ears": 0.3, "cable_exit_block": 0.5}, "links_shrink_mm": 0.0},
        "parts": manifest_parts,
    }
    graph = _mating_graph(chain, servo_meta, kit, facts, include_base_mount)
    paths = {"parts_manifest.json": dump_json(manifest, os.path.join(out_dir, "parts_manifest.json")),
             "mating_graph.json": dump_json(graph, os.path.join(out_dir, "mating_graph.json")),
             "meshes": os.path.join(out_dir, "meshes")}
    return {"parts_manifest": manifest, "mating_graph": graph, "paths": paths}


def _mating_graph(chain, servo_meta, kit, facts, include_base_mount):
    horn_pt = np.asarray(servo_meta["horn"]["axis_point"], float)
    T_horn = pose(horn_pt, [0.0, 0.0, 0.0])            # horn frame in the servo frame (z = axis)
    ear_depth = float(servo_meta["ears"][0]["hole"]["underside_z"])
    boss_h = float(servo_meta["horn"]["boss"]["height"])
    T0 = chain.fk(np.zeros(len(chain.joints)))
    mates = []
    seat_of = {}
    for sid, carrier, Tocc in chain.all_occurrences():
        mid = "seat_{}".format(sid)
        seat_of[sid] = mid
        mates.append({
            "id": mid, "kind": "sleeve_seat", "part_a": carrier, "part_b": sid,
            "frame_a_in_a": pose_to_dict(Tocc, 1.0, 6, 9), "frame_b_in_b": pose_to_dict(np.eye(4), 1.0, 6, 9),
            "insertion_axis_in_a": round_list(Tocc[:3, :3] @ np.array([0.0, 0.0, -1.0]), 9),
            "insertion_depth_mm": ear_depth, "tol_pos_mm": MATE_TOL_POS_MM, "tol_ang_deg": MATE_TOL_ANG_DEG,
            "predicate": "full 6-DoF pose of the servo frame in the carrier within tol_pos / tol_ang",
            "weight": MATE_WEIGHTS["sleeve_seat"], "requires": [],
        })
    for j in chain.joints:
        horn_link = chain.horn_side_link(j)
        frames_b = {}
        yaws = {}
        for sid in j.driven_by:
            carrier, Tocc = chain.occurrence(sid)
            if carrier is None:
                raise ValueError("joint {} driven by unknown occurrence {}".format(j.name, sid))
            T_servo_w = T0[carrier] @ Tocc
            T_b = inv(T0[horn_link]) @ T_servo_w @ T_horn        # horn frame expressed in the horn-side link
            frames_b[sid] = pose_to_dict(T_b, 1.0, 6, 9)
            yaws[sid] = round(float(np.degrees(matrix_to_rpy(T_b[:3, :3])[2])), 6)
        first = j.driven_by[0]
        mate = {
            "id": "horn_{}".format(j.name), "kind": "horn_mate", "joint": j.name,
            "part_a": first, "part_b": horn_link, "servos": list(j.driven_by),
            "frame_a_in_a": pose_to_dict(T_horn, 1.0, 6, 9), "frame_b_in_b": frames_b[first],
            "insertion_axis_in_a": [0.0, 0.0, -1.0], "insertion_depth_mm": boss_h,
            "tol_pos_mm": MATE_TOL_POS_MM, "tol_ang_deg": MATE_TOL_ANG_DEG, "axial_tol_mm": 1.0,
            "yaw_modulo_deg": 90, "design_yaw_deg": yaws[first],
            "predicate": "horn axis vs frame_b z within tol_ang (same direction), axis line distance <= tol_pos, axial offset <= axial_tol, yaw within tol_ang of design_yaw modulo 90 deg (quadrant recorded; A3 writes joint_zero_offset_rad)",
            "weight": MATE_WEIGHTS["horn_mate"], "requires": [seat_of[s] for s in j.driven_by],
        }
        if len(j.driven_by) > 1:
            mate["frames_b_in_b"] = frames_b
            mate["design_yaw_deg_per_servo"] = yaws
            mate["dual"] = "one horn_mate evaluated against the MEAN axis of the two seated horns (CHAIN_PLAN 2.5, 05 B6)"
        mates.append(mate)
    # gripper kit: fixed jaw seat + moving jaw horn mate on S6
    s6 = kit["joint"]["driven_by"]
    carrier6, T_S6 = chain.occurrence(s6)
    if carrier6 is None:
        raise ValueError("assembly.json has no {} occurrence for the gripper kit".format(s6))
    fixed = [p for p in kit["parts"] if p["name"] == "jaw_fixed"][0]
    moving = [p for p in kit["parts"] if p["name"] == "jaw_moving"][0]
    R6 = T_S6[:3, :3]
    mates.append({
        "id": "seat_jaw_fixed", "kind": "part_seat", "part_a": carrier6, "part_b": "jaw_fixed",
        "frame_a_in_a": pose_to_dict(T_S6 @ pose_from_dict(fixed["frame_in_servo"]), 1.0, 6, 9), "frame_b_in_b": pose_to_dict(np.eye(4), 1.0, 6, 9),
        "insertion_axis_in_a": round_list(R6 @ np.asarray(fixed["seat"]["insertion_axis_in_servo"], float), 9),
        "insertion_depth_mm": fixed["seat"]["insertion_depth_mm"], "tol_pos_mm": MATE_TOL_POS_MM, "tol_ang_deg": MATE_TOL_ANG_DEG,
        "predicate": "full 6-DoF pose of the jaw_fixed frame in the carrier within tol_pos / tol_ang",
        "weight": MATE_WEIGHTS["part_seat"], "requires": [seat_of[s6]],
    })
    T_b_jaw = inv(pose_from_dict(moving["frame_in_servo"])) @ T_horn   # = identity for the kit
    mates.append({
        "id": "horn_jaw", "kind": "horn_mate", "joint": kit["joint"]["name"], "part_a": s6, "part_b": "jaw_moving", "servos": [s6],
        "frame_a_in_a": pose_to_dict(T_horn, 1.0, 6, 9), "frame_b_in_b": pose_to_dict(T_b_jaw, 1.0, 6, 9),
        "insertion_axis_in_a": list(moving["seat"]["insertion_axis_in_servo"]), "insertion_depth_mm": moving["seat"]["insertion_depth_mm"],
        "tol_pos_mm": MATE_TOL_POS_MM, "tol_ang_deg": MATE_TOL_ANG_DEG, "axial_tol_mm": moving["seat"]["axial_tol_mm"],
        "yaw_modulo_deg": moving["seat"]["yaw_modulo_deg"], "design_yaw_deg": round(float(np.degrees(matrix_to_rpy(T_b_jaw[:3, :3])[2])), 6),
        "predicate": "as horn_mate; the jaw joint's zero offset is compensated in robot.xml like the arm joints",
        "weight": MATE_WEIGHTS["horn_mate"], "requires": [seat_of[s6]],
    })
    if include_base_mount:
        mates.append({
            "id": "base_mount", "kind": "base_mount", "part_a": "fixture", "part_b": chain.root,
            "frame_a_in_a": pose_to_dict(np.eye(4), 1.0, 6, 9), "frame_b_in_b": pose_to_dict(np.eye(4), 1.0, 6, 9),
            "insertion_axis_in_a": [0.0, 0.0, -1.0], "insertion_depth_mm": 5.0,
            "tol_pos_mm": MATE_TOL_POS_MM, "tol_ang_deg": MATE_TOL_ANG_DEG, "yaw_modulo_deg": 90,
            "predicate": "base frame vs fixture frame within tol (yaw modulo 90 deg: square hole pattern)",
            "weight": MATE_WEIGHTS["base_mount"], "requires": [], "enabled_from_rung": 4, "in_default_scope": False,
        })
    default_scope = [m["id"] for m in mates if m.get("in_default_scope", True)]
    prefix = []
    for j in chain.joints[:2]:
        prefix += [seat_of[s] for s in j.driven_by]
        prefix.append("horn_{}".format(j.name))
    # order the prefix as CHAIN_PLAN 4.3 lists it: seat(s) of joint 1, horn 1, seats of joint 2, horn 2
    total_w = sum(m["weight"] for m in mates if m["id"] in default_scope)
    return {
        "handoff_version": HANDOFF_VERSION, "adapter": "A2 hardware_design_to_assembly",
        "units": {"length": "mm", "angle": "rad"},
        "conventions": {
            "frame_a_in_a / frame_b_in_b": "the two frames that coincide when the mate is achieved, expressed in part_a's and part_b's own frames (URDF semantics of assembly.json)",
            "insertion_axis_in_a": "unit direction of part_b's motion during the final insertion, in part_a's frame",
            "servo_frame": "servo_standin.json `frame`; horn frame = servo frame translated to the horn axis point, z = horn axis",
            "requires": "mates that must be achieved before this one is evaluated (a link welded to a loose servo is unphysical)",
        },
        "latch": LATCH,
        "mates": mates,
        "scopes": {"rung1_prefix": prefix, "default": default_scope},
        "totals": {"mates_default_scope": len(default_scope), "weight_default_scope": round(total_w, 6)},
    }


# ============================================================================= A3
def hardware_design_to_robot_description(design_dir, bom_json, gripper_kit, out_dir, servo_standin,
                                         density=PLA_DENSITY_KG_M3, max_hulls=16, coacd_threshold=0.05):
    """A3: hardware-design (or assembly `as_built/`) -> the robot-description givens, SI metres.

    `design_dir` holds parts/ + assembly.json (mm); an as_built/assembly.json may carry
    `joints[].joint_zero_offset_rad` and `joints[].as_built_origin` (kept, converted). Convention
    (CHAIN_PLAN 6 Q14, 2026-09-27): the offset is servo calibration data (design angle at servo
    zero), `origin` stays the design child frame at q = 0 (`as_built_origin == origin`, welds snap),
    `Chain.fk` and the reference robot.xml ignore it (no `ref`; qpos == design angle), so A3 of an
    as_built/ yields the same robot.xml as A3 of the design; `mjcf_reference.py` docstring. Writes:
      assembly.json        metres; links[].mass_properties {mass_kg, com_m, inertia_kg_m2} from
                           solid x density + servo boxes (arm_design_check._inertia_of_link :878);
                           joints[].driven_by as a list; joints[].joint_zero_offset_rad; a `gripper`
                           block (kit composed with the carrier's S6 occurrence) and the `tcp`
      meshes/*.stl         visual meshes in metres (links, servo_standin, jaw_fixed, jaw_moving)
      hulls/*.stl          <= max_hulls CoACD convex pieces per link and jaw, metres
      actuator_facts.json  per joint: count, servo ids, catalogue id, stall (each / total), speed, mass
      gripper.json         kit in the carrier frame, metres (03 section 1.2 shape)
    Returns {"assembly": ..., "actuator_facts": ..., "gripper": ..., "paths": {...}}.
    """
    parts_dir = os.path.join(design_dir, "parts")
    asm_path = os.path.join(design_dir, "assembly.json")
    asm = load_json(asm_path)
    facts = actuator_facts_from_bom(bom_json)
    servo_json, servo_step = _resolve_servo(servo_standin)
    servo_meta = load_json(servo_json)
    kit = _resolve_kit(gripper_kit)
    chain = Chain(asm, tool_from_kit(kit))
    mesh_dir = os.path.join(out_dir, "meshes")
    hull_dir = os.path.join(out_dir, "hulls")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(hull_dir, exist_ok=True)

    solids = _link_solids(parts_dir, chain)
    links_out = []
    for name in chain.order:
        link = chain.links[name]
        mesh = solids[name][1]
        m, c, I, m_link = _link_mass_properties(link, mesh, facts, servo_meta, density)
        export_stl(mesh, os.path.join(mesh_dir, name + ".stl"), scale=MM)
        hulls = []
        for i, piece in enumerate(convex_pieces(mesh, max_hulls, coacd_threshold)):
            hulls.append("hulls/{}_{:02d}.stl".format(name, i))
            export_stl(piece, os.path.join(out_dir, hulls[-1]), scale=MM)
        entry = {
            "name": name, "part_file": link.part_file, "mesh": "meshes/{}.stl".format(name), "hulls": hulls,
            "servo_occurrences": [{"servo_id": sid, **pose_to_dict(T, MM, 12)} for sid, T in link.occurrences],
            "mass_properties": _mass_block(m, c, I, MM),
            "printed_mass_kg": round(float(m_link), 12),
            "solid_volume_m3": round(float(mesh.volume) * 1e-9, 15),
        }
        if link.mass_override is not None:
            entry["mass_override_kg"] = float(link.mass_override)
        links_out.append(entry)
    servo_mesh = tessellate(load_step_solid(servo_step))
    export_stl(servo_mesh, os.path.join(mesh_dir, "servo_standin.stl"), scale=MM)

    joints_out = []
    actuator_facts = {"handoff_version": HANDOFF_VERSION, "adapter": "A3 hardware_design_to_robot_description",
                      "source": "component_catalogue.json copied through bom.json (identity check C0 verifies the copy)",
                      "units": {"torque": "N*m", "speed": "rad/s", "mass": "kg", "voltage": "V"}, "joints": {}}
    for j in chain.joints:
        ids = list(j.driven_by)
        stall_each = [facts[s]["stall_torque_nm"] for s in ids]
        entry = {
            "name": j.name, "type": "revolute", "parent_link": j.parent, "child_link": j.child,
            "origin": pose_to_dict(j.origin, MM, 12), "axis": round_list(j.axis, 12),
            "limits": {"lower": j.lower, "upper": j.upper}, "driven_by": ids,
            "joint_zero_offset_rad": float(j.spec.get("joint_zero_offset_rad", 0.0)),
        }
        if "as_built_origin" in j.spec:
            entry["as_built_origin"] = pose_to_dict(pose_from_dict(j.spec["as_built_origin"]), MM, 12)
        joints_out.append(entry)
        actuator_facts["joints"][j.name] = {
            "servo_ids": ids, "count": len(ids), "catalogue_id": facts[ids[0]]["catalogue_id"],
            "standin_variant": facts[ids[0]]["standin_variant"],
            "stall_torque_nm_each": stall_each[0], "stall_torque_nm_total": round(sum(stall_each), 9),
            "no_load_speed_rad_s": facts[ids[0]]["no_load_speed_rad_s"],
            "mass_kg_each": facts[ids[0]]["mass_kg"], "voltage_v": facts[ids[0]]["voltage_v"],
            "forcerange_rule": "robot.xml act_{}: forcerange within [-count x stall, +count x stall] (C1)".format(j.name),
        }
    # gripper kit on the S6 carrier
    s6 = kit["joint"]["driven_by"]
    carrier6, T_S6 = chain.occurrence(s6)
    gripper = kit_in_carrier(kit, T_S6, carrier6, scale=MM)
    for part in kit["parts"]:
        mesh = tessellate(load_step_solid(os.path.join(kit["_dir"], part["part_file"])))
        export_stl(mesh, os.path.join(mesh_dir, part["name"] + ".stl"), scale=MM)
        hulls = []
        for i, piece in enumerate(convex_pieces(mesh, max_hulls, coacd_threshold)):
            hulls.append("hulls/{}_{:02d}.stl".format(part["name"], i))
            export_stl(piece, os.path.join(out_dir, hulls[-1]), scale=MM)
        key = "fixed_jaw" if part["name"] == "jaw_fixed" else "moving_jaw"
        gripper[key]["mesh"] = "meshes/{}.stl".format(part["name"])
        gripper[key]["hulls"] = hulls
    as_built_gripper = asm.get("gripper") if isinstance(asm.get("gripper"), dict) else {}
    gripper["joint"]["joint_zero_offset_rad"] = float(as_built_gripper.get("joint_zero_offset_rad", 0.0))
    actuator_facts["joints"][kit["joint"]["name"]] = {
        "servo_ids": [s6], "count": 1, "catalogue_id": facts[s6]["catalogue_id"], "standin_variant": facts[s6]["standin_variant"],
        "stall_torque_nm_each": facts[s6]["stall_torque_nm"], "stall_torque_nm_total": facts[s6]["stall_torque_nm"],
        "no_load_speed_rad_s": facts[s6]["no_load_speed_rad_s"], "mass_kg_each": facts[s6]["mass_kg"], "voltage_v": facts[s6]["voltage_v"],
        "forcerange_rule": "robot.xml act_jaw: forcerange within [-stall, +stall] (C1)",
    }
    tcp = gripper["tcp"]
    assembly_si = {
        "handoff_version": HANDOFF_VERSION, "adapter": "A3 hardware_design_to_robot_description",
        "units": {"length": "m", "angle": "rad", "mass": "kg", "inertia": "kg*m^2"},
        "source": {"design_dir": os.path.basename(os.path.abspath(design_dir)), "design_digest": design_digest(parts_dir, asm_path),
                   "servo_standin_step_sha256": sha256_file(servo_step), "gripper_kit": kit["kit_id"],
                   "as_built": "as_built_origin" in asm["joints"][0]},
        "conventions": {"frames": "link frame = STEP part frame; joint origin = child frame in parent at q = 0; axis in the child frame; rpy fixed-axis XYZ (CHAIN_PLAN 2.1)",
                        "mass_properties": "solid x {} kg/m^3 (or the larger mass_override_kg) + one box per servo occurrence with the occurrence's catalogue mass at the stand-in centre of mass; inertia about the CoM in link axes (arm_design_check._inertia_of_link :878)".format(density),
                        "joint_zero_offset_rad": "angle about the joint axis between the as-built link and the designed link when the servo horn is at its own zero (0 for a design, k x pi/2 from an assembly record); robot.xml compensates it (C-A)"},
        "root_link": chain.root,
        "links": links_out,
        "joints": joints_out,
        "servo_standin": {"mesh": "meshes/servo_standin.stl", "body_size_m": round_list(np.asarray(servo_meta["body"]["size"]) * MM, 9),
                          "horn_axis_point_m": round_list(np.asarray(servo_meta["horn"]["axis_point"]) * MM, 9),
                          "centre_of_mass_m": round_list(np.asarray(servo_meta.get("centre_of_mass", [0, 0, servo_meta["body"]["size"][2] / 2])) * MM, 9)},
        "gripper": gripper,
        "tcp": {"parent_link": tcp["parent_link"], "xyz": tcp["xyz"], "rpy": tcp["rpy"], "convention": tcp["convention"]},
    }
    paths = {"assembly.json": dump_json(assembly_si, os.path.join(out_dir, "assembly.json")),
             "actuator_facts.json": dump_json(actuator_facts, os.path.join(out_dir, "actuator_facts.json")),
             "gripper.json": dump_json(gripper, os.path.join(out_dir, "gripper.json")),
             "meshes": mesh_dir, "hulls": hull_dir}
    return {"assembly": assembly_si, "actuator_facts": actuator_facts, "gripper": gripper, "paths": paths}


# ============================================================================= A4
IK_INTERFACE_PY = '''"""ik_interface.py: the control stage's typed contract (adapter A4 output; CHAIN_PLAN 2.4).

`solve(model, target, q0) -> q[5] | None`: joint angles of the five arm joints (ARM_JOINTS order,
inside model.jnt_range) that place the `tcp` site at `target["pos"]` (metres, base frame) with
the site's +z axis along `target["approach"]` (unit vector), optionally with a full orientation
`target["quat"]` (wxyz) or a roll about the approach axis `target["roll_rad"]`; `None` when the
target is unreachable (false positives are counted). The grader compiles the SUBMITTED robot.xml
and passes the model object (identity check C2); ik.py may not load any other model.
"""
from typing import Optional, Protocol, TypedDict

import mujoco
import numpy as np

ARM_JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
JAW_JOINT = "jaw"
TCP_SITE = "tcp"
HANDOFF_VERSION = "%(version)s"


class Target(TypedDict, total=False):
    pos: list          # [3] metres, base (world) frame
    approach: list     # [3] unit vector, world frame: desired +z of the tcp site
    roll_rad: float    # optional roll about the approach axis
    quat: list         # optional full orientation, wxyz


def arm_qpos_indices(model):
    return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]


def fk(model, data, q, jaw=None):
    """(pos[3], rotmat[3x3]) of the tcp site for the five arm angles q (jaw held where it is)."""
    idx = arm_qpos_indices(model)
    data.qpos[idx] = np.asarray(q, float)
    if jaw is not None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JAW_JOINT)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = float(jaw)
    mujoco.mj_kinematics(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


class Solver(Protocol):
    def __call__(self, model: mujoco.MjModel, target: Target, q0: np.ndarray) -> Optional[np.ndarray]: ...
'''


def robot_description_to_control(robot_xml, scene_xml, out_dir, n_targets=200, seed=20260926,
                                 near_singular_ratio=0.02, jaw_home=None, collision_free=False):
    """A4: robot.xml + scene.xml -> the control stage's givens (CHAIN_PLAN 2.3 A4, 4.5).

    Compiles the model, writes `model.mjb` (the binary that crosses the UID boundary, 05 A9),
    `interface.yaml` (joint order, ranges, actuators, tcp site, jaw), `ik_interface.py` and
    `practice_targets.json` (FK of `n_targets` configurations drawn uniformly inside the joint
    ranges by numpy's PCG64 from `seed`; each target {pos, approach, quat, reachable: true,
    near_singular}; q_true withheld). Deterministic for a given (model, seed). With
    `collision_free=True` (the control task's rung-1 givens, CHAIN_PLAN 4.5: hidden targets are FK of
    collision-free configurations) configurations with any active contact after `mj_forward` (self
    or floor) are rejected and redrawn, so practice and hidden targets share one distribution; the
    default keeps the unfiltered output. The drawn q are MuJoCo qpos == design joint angles (Q14: the
    robot.xml carries no `ref`), so every target equals `Chain.fk(q)` of the A3 assembly.json
    composed with its `tcp` (tests/test_zero_offset_convention.py); a servo-zero offset of an as-built
    arm never enters the control hand-off (servo command = sign x (qpos - offset) is the assembly
    stage's calibration fact).
    """
    import mujoco
    import yaml
    model = mujoco.MjModel.from_xml_path(scene_xml)
    os.makedirs(out_dir, exist_ok=True)
    jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    missing = [j for j in ARM_JOINTS + [JAW_JOINT] if j not in jn]
    if missing:
        raise ValueError("robot model lacks the contract joints {}".format(missing))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if sid < 0:
        raise ValueError("robot model has no site 'tcp'")
    joints = []
    for name in ARM_JOINTS + [JAW_JOINT]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joints.append({"name": name, "qpos_index": int(model.jnt_qposadr[jid]),
                       "range_rad": round_list(model.jnt_range[jid], 9),
                       "type": "hinge" if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE else str(int(model.jnt_type[jid]))})
    actuators = []
    for i in range(model.nu):
        trn = int(model.actuator_trnid[i, 0])
        actuators.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                          "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trn),
                          "ctrlrange": round_list(model.actuator_ctrlrange[i], 9),
                          "forcerange": round_list(model.actuator_forcerange[i], 9)})
    mjb_path = os.path.join(out_dir, "model.mjb")
    mujoco.mj_saveModel(model, mjb_path, None)
    interface = {
        "handoff_version": HANDOFF_VERSION, "adapter": "A4 robot_description_to_control",
        "units": {"length": "m", "angle": "rad"},
        "model": {"robot_xml": os.path.basename(robot_xml), "scene_xml": os.path.basename(scene_xml), "mjb": "model.mjb",
                  "mjb_sha256": sha256_file(mjb_path), "nq": int(model.nq), "nu": int(model.nu)},
        "arm_joints": ARM_JOINTS, "jaw_joint": JAW_JOINT, "joints": joints, "actuators": actuators,
        "tcp_site": "tcp", "tcp_convention": "+z = approach (palm -> jaw tips), +x = jaw closing direction",
        "solve_signature": "solve(model: mujoco.MjModel, target: Target, q0: np.ndarray[5]) -> np.ndarray[5] | None",
        "success": {"position_m": 0.002, "approach_axis_deg": 1.0, "quat_geodesic_deg": 1.0, "limits_slack_rad": 1e-6},
    }
    with open(os.path.join(out_dir, "interface.yaml"), "w") as f:
        yaml.safe_dump(interface, f, sort_keys=False)
    with open(os.path.join(out_dir, "ik_interface.py"), "w") as f:
        f.write(IK_INTERFACE_PY % {"version": HANDOFF_VERSION})
    # practice targets
    data = mujoco.MjData(model)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    idx = [j["qpos_index"] for j in joints[:5]]
    lo = np.array([j["range_rad"][0] for j in joints[:5]])
    hi = np.array([j["range_rad"][1] for j in joints[:5]])
    jaw_idx = joints[5]["qpos_index"]
    jaw_val = float(jaw_home) if jaw_home is not None else 0.5 * (joints[5]["range_rad"][0] + joints[5]["range_rad"][1])
    targets = []
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    dof_idx = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS]
    n_sing = 0
    n_rejected = 0
    while len(targets) < int(n_targets):
        q = lo + rng.random(5) * (hi - lo)
        data.qpos[:] = 0.0
        data.qpos[idx] = q
        data.qpos[jaw_idx] = jaw_val
        if collision_free:
            mujoco.mj_forward(model, data)
            if data.ncon > 0:
                n_rejected += 1
                continue
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        sv = np.linalg.svd(jacp[:, dof_idx], compute_uv=False)
        near = bool(sv[-1] / max(sv[0], 1e-12) < near_singular_ratio)
        n_sing += int(near)
        R = data.site_xmat[sid].reshape(3, 3)
        targets.append({"pos": round_list(data.site_xpos[sid], 9), "approach": round_list(R[:, 2], 9),
                        "quat": round_list(quat_from_matrix(R), 9), "reachable": True, "near_singular": near})
    doc = {"handoff_version": HANDOFF_VERSION, "adapter": "A4 robot_description_to_control", "units": {"length": "m"},
           "generation": "FK of configurations drawn uniformly inside the joint ranges (numpy PCG64, seed {}), jaw at {:.6f}; q_true withheld".format(seed, jaw_val),
           "count": len(targets), "near_singular_count": n_sing, "targets": targets}
    if collision_free:
        doc["generation"] += "; configurations with any contact (self or floor) after mj_forward rejected ({} redrawn)".format(n_rejected)
        doc["collision_free"] = True
        doc["rejected_count"] = n_rejected
    dump_json(doc, os.path.join(out_dir, "practice_targets.json"))
    return {"interface": interface, "paths": {"interface.yaml": os.path.join(out_dir, "interface.yaml"),
                                              "ik_interface.py": os.path.join(out_dir, "ik_interface.py"),
                                              "practice_targets.json": os.path.join(out_dir, "practice_targets.json"),
                                              "model.mjb": mjb_path}}


# ============================================================================= A5
def control_and_description_to_environment(robot_xml, ik_py, out_dir, robot_root="base"):
    """A5: robot.xml + ik.py -> `env_givens/` for the environment stage (CHAIN_PLAN 2.3 A5, 2.6 C3).

    Writes robot/robot.xml + robot/assets/ (verbatim copies of every referenced mesh file),
    robot/scene.xml when it sits next to robot.xml, robot_canonical.json (the canonical form
    the C3 identity check compares against, with its sha256), ik/ik.py (+ ik_interface.py when
    present next to it), action_space.json (6 absolute joint position targets in ctrlrange,
    CHAIN_PLAN 4.7) and the privileged observation contract.
    """
    from common.mjcf_canonical import canonical_robot, canonical_sha256, referenced_files
    os.makedirs(os.path.join(out_dir, "robot"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ik"), exist_ok=True)
    shutil.copyfile(robot_xml, os.path.join(out_dir, "robot", "robot.xml"))
    scene_src = os.path.join(os.path.dirname(robot_xml), "scene.xml")
    if os.path.isfile(scene_src):
        shutil.copyfile(scene_src, os.path.join(out_dir, "robot", "scene.xml"))
    for src in referenced_files(robot_xml):
        dst = os.path.join(out_dir, "robot", os.path.relpath(src, os.path.dirname(robot_xml)))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    canon = canonical_robot(robot_xml, robot_root)
    canon_doc = {"handoff_version": HANDOFF_VERSION, "adapter": "A5 control_and_description_to_environment",
                 "robot_root_body": robot_root, "canonical_sha256": canonical_sha256(canon), "canonical": canon,
                 "rule": "environment/scene.xml must contain this subtree byte-identically after canonicalisation (attribute order free, numbers rounded to 1e-9, mesh files by basename); actuators unchanged; root body at the given mount frame (C3)"}
    dump_json(canon_doc, os.path.join(out_dir, "robot_canonical.json"))
    if ik_py:
        shutil.copyfile(ik_py, os.path.join(out_dir, "ik", "ik.py"))
        iface = os.path.join(os.path.dirname(ik_py), "ik_interface.py")
        if os.path.isfile(iface):
            shutil.copyfile(iface, os.path.join(out_dir, "ik", "ik_interface.py"))
    import mujoco
    model = mujoco.MjModel.from_xml_path(robot_xml)
    acts = []
    for i in range(model.nu):
        acts.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                     "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])),
                     "ctrlrange": round_list(model.actuator_ctrlrange[i], 9)})
    action_space = {
        "handoff_version": HANDOFF_VERSION, "adapter": "A5 control_and_description_to_environment",
        "action": {"size": len(acts), "semantics": "absolute position targets for the actuators below, in their ctrlrange, actuator order",
                   "units": "rad", "rate_limit_rad_per_step": 0.325, "control_hz": 20, "episode_steps": 600, "actuators": acts},
        "observation": {"tier": "privileged", "keys": {"qpos": 6, "qvel": 6, "tcp_pose": "7 (pos xyz + quat wxyz, metres)",
                                                        "object_poses": "{name: [7]}", "goal_region": "6 (AABB min xyz, max xyz, metres)"}},
        "env_api": {"class": "gymnasium.Env", "methods": ["reset(seed) -> obs, info", "step(action) -> obs, reward, terminated, truncated, info",
                                                           "set_state(state)", "get_state()", "success() -> bool"]},
        "ik": {"file": "ik/ik.py" if ik_py else None, "interface": "ik/ik_interface.py"},
    }
    dump_json(action_space, os.path.join(out_dir, "action_space.json"))
    return {"canonical_sha256": canon_doc["canonical_sha256"], "paths": {"robot": os.path.join(out_dir, "robot"),
            "robot_canonical.json": os.path.join(out_dir, "robot_canonical.json"), "action_space.json": os.path.join(out_dir, "action_space.json")}}


# ============================================================================= A6
WIRE_MD = """# wire.md: the behavior stage's actor protocol (adapter A6; CHAIN_PLAN.md section 2.4 "Wire (C4)")

Identical to `tasks/control/drone_hover/instruction.md`: `run.sh` starts one long-lived process
that speaks JSON lines on stdin / stdout. The grader owns the simulator, spawns the actor as the
unprivileged user with `runuser` and a per-step timeout (`drone_hover/verify/verify.py`, `Solver`),
applies actions only to the robot's actuators and recomputes success from its own state (C4).

1. print a single line `READY` once loaded (library banners before it are tolerated);
2. `{"type": "reset", "seed": k, "task": {...}}` -> `{"ok": true}`  (clear recurrent state; `task`
   carries the goal region and object names of the episode as `task.yaml` declares them);
3. `{"type": "act", "obs": {...}, "probe": true?}` -> `{"action": [a1, ..., a6]}`  (6 absolute joint
   position targets, radians, actuator order and `ctrlrange` of `action_space.json`; 20 Hz, 600
   steps = 30 s; rate limit 0.325 rad/step; requests may carry a grader step index `t` and a
   repeated `t` must be answered from the observation it carries);
4. `{"type": "close"}` -> `{"ok": true}`, then exit.

Keep stdout clean: only `READY` and protocol replies; diagnostics to stderr. One process serves a
few episodes (each opened by its own `reset`; the behavior grader gives one process three episodes
of three different hidden seeds in a hidden order), then `close`. Deadlines are absolute (READY
within its timeout whatever is printed first; every reply within the per-request timeout).

Probes: repeats 1-2 of every hidden seed carry grader-owned perturbations at hidden moments (the
untouched mug displaced 1-3 cm the first time the tcp approaches it, actions held for 5-12 steps,
observations stale for 3-8 steps); among the probed repeats whose clean twin succeeded, the share
that still succeeds must be >= 0.5 (gate; 0 when no clean repeat succeeded). Some `act` requests
additionally carry `"probe": true` (same step index, perturbed observation): answer them from the
observation they carry WITHOUT advancing your state; the reply is never applied and the grader only
records whether it differs (stat obs_dependence_fraction). A policy that ignores its input, replays a
recording or returns a constant fails the perturbations. `env.success()` is never consulted for credit.

Observation (privileged tier): `qpos[6]`, `qvel[6]`, `tcp_pose[7]` (xyz + quat wxyz, metres),
`object_poses {name: [7]}`, `goal_region[6]` (AABB min / max, metres). Units metres / radians.
"""


def environment_to_behavior(env_pkg_dir, out_dir, exclude=("oracle_demo.py", "trajectories", "oracle", "__pycache__")):
    """A6: the environment package -> the behavior stage's givens (CHAIN_PLAN 2.3 A6, 2.4, 4.7).

    Copies the package without the oracle material (`oracle_demo.py`, `trajectories/`, `oracle/`),
    writes `wire.md` (the stdio protocol) and `obs_action_schema.json` (observation / action
    shapes: from the package's task.yaml `observation` / `action` blocks when present, else the
    chain contract's privileged defaults). No code of the package is executed.
    """
    import yaml
    env_out = os.path.join(out_dir, "env")
    if os.path.isdir(env_out):
        shutil.rmtree(env_out)

    def _ignore(_dir, names):
        return [n for n in names if n in exclude]
    shutil.copytree(env_pkg_dir, env_out, ignore=_ignore)
    task_yaml = os.path.join(env_pkg_dir, "task.yaml")
    task = {}
    if os.path.isfile(task_yaml):
        with open(task_yaml) as f:
            task = yaml.safe_load(f) or {}
    obs = task.get("observation") or {"tier": "privileged", "qpos": 6, "qvel": 6, "tcp_pose": 7,
                                       "object_poses": "{name: [7]}", "goal_region": 6}
    act = task.get("action") or {"size": 6, "semantics": "absolute joint position targets (rad) in ctrlrange, actuator order",
                                  "control_hz": 20, "episode_steps": 600, "rate_limit_rad_per_step": 0.325}
    schema = {"handoff_version": HANDOFF_VERSION, "adapter": "A6 environment_to_behavior",
              "observation": obs, "action": act,
              "reset_payload": {"seed": "int", "task": task.get("task_payload_keys", ["objects", "goal_region", "predicate"])},
              "source": "task.yaml observation / action blocks" if task.get("observation") else "chain contract defaults (CHAIN_PLAN 2.4)",
              "excluded_from_package": list(exclude)}
    with open(os.path.join(out_dir, "wire.md"), "w") as f:
        f.write(WIRE_MD)
    dump_json(schema, os.path.join(out_dir, "obs_action_schema.json"))
    return {"schema": schema, "paths": {"env": env_out, "wire.md": os.path.join(out_dir, "wire.md"),
                                        "obs_action_schema.json": os.path.join(out_dir, "obs_action_schema.json")}}
