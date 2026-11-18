"""The gripper kit as a bank asset: loading, composition with a carrier, MJCF snippet.

`gripper_kit/gripper_kit.json` expresses every kit frame in the S6 servo frame
(`servo_standin.json` `frame`), so the kit is carrier-agnostic: adapters A2 / A3 compose it
with the carrier's S6 occurrence pose `T_S6` (servo frame in the carrier link frame, mm) taken
from `assembly.json` (CHAIN_PLAN.md sections 2.2 "gripper kit", 2.3 A2 / A3, D9).
"""

import json
import os

import numpy as np

from .frames import apply, pose_from_dict, pose_to_dict, quat_from_matrix, round_list

__all__ = ["load_kit", "kit_in_carrier", "mjcf_snippet", "KIT_DIR"]

KIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_kit")


def load_kit(kit_dir=KIT_DIR):
    """The kit JSON with absolute part / mesh paths added (`_dir`, `_part_paths`, `_mesh_paths`)."""
    with open(os.path.join(kit_dir, "gripper_kit.json")) as f:
        kit = json.load(f)
    kit["_dir"] = os.path.abspath(kit_dir)
    kit["_part_paths"] = dict((p["name"], os.path.join(kit_dir, p["part_file"])) for p in kit["parts"])
    kit["_mesh_paths"] = dict((p["name"], os.path.join(kit_dir, p["mesh"])) for p in kit["parts"])
    return kit


def _part(kit, name):
    for p in kit["parts"]:
        if p["name"] == name:
            return p
    raise KeyError(name)


def kit_in_carrier(kit, T_S6, carrier_name, scale=1.0):
    """Kit frames re-expressed in the carrier link frame (mm x `scale`).

    Returns a `gripper.json`-shaped dict (03-atomic-tasks-design.md section 1.2): `fixed_jaw`
    pose, the `jaw` joint (origin = child frame in the carrier at q = 0, axis in the child
    frame), `tcp` pose, pads, masses. `T_S6` is the S6 occurrence pose in the carrier frame.
    """
    fixed = _part(kit, "jaw_fixed")
    moving = _part(kit, "jaw_moving")
    T_fixed = T_S6 @ pose_from_dict(fixed["frame_in_servo"])
    T_joint = T_S6 @ pose_from_dict(kit["joint"]["origin_in_servo"])
    T_tcp = T_S6 @ pose_from_dict(kit["tcp"]["in_servo"])
    unit = "m" if abs(scale - 1e-3) < 1e-12 else ("mm" if scale == 1.0 else "mm*{}".format(scale))
    out = {
        "handoff_version": kit.get("handoff_version"),
        "kit_id": kit["kit_id"],
        "units": {"length": unit, "angle": "rad", "mass": "kg"},
        "carrier_link": carrier_name,
        "servo_id": kit["joint"]["driven_by"],
        "fixed_jaw": {
            "name": "jaw_fixed", "part_file": fixed["part_file"], "mesh": fixed["mesh"],
            "parent_link": carrier_name, "origin": pose_to_dict(T_fixed, scale),
            "mass_properties": _mass_props(fixed, scale),
            "pad": _pad(fixed, scale),
        },
        "joint": {
            "name": kit["joint"]["name"], "type": kit["joint"]["type"], "parent_link": carrier_name,
            "child_link": "jaw_moving", "origin": pose_to_dict(T_joint, scale),
            "axis": list(kit["joint"]["axis_in_child"]),
            "limits": dict(kit["joint"]["limits_rad"]), "driven_by": kit["joint"]["driven_by"],
            "positive_direction": kit["joint"]["positive_direction"],
        },
        "moving_jaw": {
            "name": "jaw_moving", "part_file": moving["part_file"], "mesh": moving["mesh"],
            "mass_properties": _mass_props(moving, scale),
            "pad": _pad(moving, scale),
        },
        "tcp": {"parent_link": carrier_name, **pose_to_dict(T_tcp, scale),
                "convention": kit["tcp"]["convention"]},
        "pad_contact_geoms": [fixed["pad"]["geom_name"], moving["pad"]["geom_name"]],
        "jaw_travel": kit["jaw_travel"],
    }
    return out


def _mass_props(part, scale):
    I = np.asarray(part["inertia_kg_mm2"], float) * (scale ** 2)
    key_com = "com_m" if abs(scale - 1e-3) < 1e-12 else "com_mm"
    key_I = "inertia_kg_m2" if abs(scale - 1e-3) < 1e-12 else "inertia_kg_mm2"
    return {"mass_kg": float(part["mass_kg"]),
            key_com: round_list(np.asarray(part["com_mm"], float) * scale, 12),
            key_I: [round_list(row, 15) for row in I]}


def _pad(part, scale):
    lo = np.asarray(part["pad"]["box_min_part"], float) * scale
    hi = np.asarray(part["pad"]["box_max_part"], float) * scale
    return {"geom_name": part["pad"]["geom_name"], "box_min": round_list(lo, 12), "box_max": round_list(hi, 12),
            "normal_part": list(part["pad"]["normal_part"]), "friction": list(part["pad"].get("friction", [1.0, 0.005, 0.0001]))}


def _fmt(values, nd=9):
    return " ".join("{:.{}f}".format(v, nd).rstrip("0").rstrip(".") if v != 0 else "0" for v in values)


def mjcf_snippet(kit, T_S6, carrier_name="gripper_carrier", mesh_prefix="", hull_names=None,
                 jaw_kp=2.0, jaw_forcerange=2.942, indent="      "):
    """MJCF (metres) of the kit under the carrier body: two bodies, the `jaw` hinge, pads, `tcp`.

    `hull_names`: optional {part: [mesh names]} of collision hulls (A3 output); without them
    the visual mesh doubles as the collision geom (MuJoCo uses its convex hull).
    Returns (body_xml, asset_xml, actuator_xml) fragments.
    """
    g = kit_in_carrier(kit, T_S6, carrier_name, scale=1e-3)
    hull_names = hull_names or {}
    lines = ["{}<!-- gripper kit {} (bank asset of arb_arm_a1); frames from gripper_kit.json composed with the".format(indent, kit["kit_id"]),
             "{}     S6 occurrence of {}; units m / rad; jaw range {} .. {} rad, 0 = closed -->".format(
                 indent, carrier_name, g["joint"]["limits"]["lower"], g["joint"]["limits"]["upper"])]
    for key, body_name, joint in (("fixed_jaw", "jaw_fixed", None), ("moving_jaw", "jaw_moving", g["joint"])):
        part = g[key]
        T = g["fixed_jaw"]["origin"] if key == "fixed_jaw" else g["joint"]["origin"]
        mp = part["mass_properties"]
        I = np.asarray(mp["inertia_kg_m2"], float)
        quat = quat_from_matrix(pose_from_dict(T)[:3, :3])
        lines.append('{}<body name="{}" pos="{}" quat="{}">'.format(indent, body_name, _fmt(T["xyz"]), _fmt(quat)))
        if joint is not None:
            lines.append('{}  <joint name="{}" type="hinge" axis="{}" range="{} {}" damping="0.01" armature="0.0001" frictionloss="0.001"/>'.format(
                indent, joint["name"], _fmt(joint["axis"]), joint["limits"]["lower"], joint["limits"]["upper"]))
        lines.append('{}  <inertial pos="{}" mass="{:.9f}" fullinertia="{:.6e} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}"/>'.format(
            indent, _fmt(mp["com_m"], 12), mp["mass_kg"], I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
        lines.append('{}  <geom name="{}_visual" type="mesh" mesh="{}{}" contype="0" conaffinity="0" group="2" rgba="0.85 0.85 0.9 1"/>'.format(
            indent, body_name, mesh_prefix, body_name))
        hulls = hull_names.get(body_name)
        if hulls:
            for h in hulls:
                lines.append('{}  <geom name="{}" type="mesh" mesh="{}" group="3" rgba="0.3 0.6 0.9 0.4"/>'.format(indent, h, h))
        else:
            lines.append('{}  <geom name="{}_collision" type="mesh" mesh="{}{}" group="3" rgba="0.3 0.6 0.9 0.4"/>'.format(
                indent, body_name, mesh_prefix, body_name))
        pad = part["pad"]
        lo, hi = np.asarray(pad["box_min"]), np.asarray(pad["box_max"])
        lines.append('{}  <geom name="{}" type="box" pos="{}" size="{}" friction="{}" condim="4" rgba="0.1 0.1 0.1 1"/>'.format(
            indent, pad["geom_name"], _fmt((lo + hi) / 2, 12), _fmt((hi - lo) / 2, 12), _fmt(pad["friction"], 6)))
        lines.append("{}</body>".format(indent))
    tcp = g["tcp"]
    lines.append('{}<site name="tcp" pos="{}" quat="{}" size="0.003" rgba="1 0 0 0.5"/>'.format(
        indent, _fmt(tcp["xyz"]), _fmt(quat_from_matrix(pose_from_dict(tcp)[:3, :3]))))
    assets = ['<mesh name="{}{}" file="{}{}.stl" scale="0.001 0.001 0.001"/>'.format(mesh_prefix, n, mesh_prefix, n)
              for n in ("jaw_fixed", "jaw_moving")]
    actuator = ('<position name="act_jaw" joint="jaw" kp="{}" ctrlrange="{} {}" forcerange="-{} {}"/>'.format(
        jaw_kp, g["joint"]["limits"]["lower"], g["joint"]["limits"]["upper"], jaw_forcerange, jaw_forcerange))
    return "\n".join(lines), "\n".join(assets), actuator
