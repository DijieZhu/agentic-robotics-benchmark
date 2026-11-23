"""The kinematic content of `assembly.json` (no geometry needed).

Reproduces `arm_design_check.Chain` (arm_design_check.py:291-331) with two generalisations
the chain contract adds (CHAIN_PLAN.md section 3.2): `driven_by` is one id or a list of 1-2
occurrence ids, and servo ids follow the occurrence scheme S1, S2A, S2B, S3A, S3B, S4, S5, S6.
Works in the file's own units (mm for hardware-design files, m for the A3 output).

`joint_zero_offset_rad` (CHAIN_PLAN 6 Q14, settled 2026-09-27): `Joint.zero_offset` is read and kept
as data, and `Chain.fk` deliberately does NOT apply it. The kinematic truth of assembly.json (joint
origin, axis, child placement at q = 0) is the DESIGN pose whether the file comes from
hardware-design/ or from an assembly `as_built/` (the cell's welds snap to the design, so
`as_built_origin == origin`); the offset only says at which design angle the as-built link sits when
its servo horn reads zero (a quadrant, k x pi/2 plus the pre-snap residual). So fk(q) is the design
pose for every q, q == MuJoCo qpos of the reference robot.xml (no `ref`), and the servo command is
`theta_s = sign_s x (q - offset)` (sign_s = horn axis . joint axis per physical servo, the cell's
`servo_drive`). Full statement: `mjcf_reference.py` module docstring.
"""

import numpy as np

from .frames import apply, pose, rot_axis

__all__ = ["JOINT_ORDER", "Chain", "driven_by_list", "occurrence_ids", "servo_joint_index"]

JOINT_ORDER = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]


def driven_by_list(value):
    """`driven_by` as a list of occurrence ids (accepts the CAD schema's single string)."""
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def occurrence_ids(k, count):
    """Occurrence ids of joint k with `count` servos: 1 -> S<k>; 2 -> S<k>A, S<k>B (A1 rule)."""
    if count == 1:
        return ["S{}".format(k)]
    if count == 2:
        return ["S{}A".format(k), "S{}B".format(k)]
    raise ValueError("count must be 1 or 2, got {}".format(count))


def servo_joint_index(servo_id):
    """S1 -> 1, S2A -> 2, S6 -> 6 (the joint number encoded in an occurrence id)."""
    digits = "".join(ch for ch in servo_id[1:] if ch.isdigit())
    return int(digits)


class Link(object):
    def __init__(self, spec):
        self.name = spec["name"]
        self.part_file = spec.get("part_file")
        self.mass_override = spec.get("mass_override_kg")
        self.occurrences = [(o["servo_id"], pose(o["xyz"], o["rpy"])) for o in spec.get("servo_occurrences", [])]
        self.spec = spec


class Joint(object):
    def __init__(self, spec):
        self.name = spec["name"]
        self.parent = spec["parent_link"]
        self.child = spec["child_link"]
        self.origin = pose(spec["origin"]["xyz"], spec["origin"]["rpy"])
        a = np.asarray(spec["axis"], float)
        self.axis = a / (np.linalg.norm(a) if np.linalg.norm(a) > 0 else 1.0)
        self.lower = float(spec["limits"]["lower"])
        self.upper = float(spec["limits"]["upper"])
        self.driven_by = driven_by_list(spec["driven_by"])
        # as-built servo calibration (design angle at servo zero); never enters fk (module docstring, Q14)
        self.zero_offset = float(spec.get("joint_zero_offset_rad", 0.0))
        self.spec = spec


class Chain(object):
    """Serial chain rooted at `root_link`; FK returns the link frames in the root frame."""

    def __init__(self, spec, tool=None):
        self.spec = spec
        self.units = spec.get("units", {})
        self.links = dict((l["name"], Link(l)) for l in spec["links"])
        self.joints = [Joint(j) for j in spec["joints"]]
        self.root = spec["root_link"]
        self.tool = tool   # {servo_id, point (servo frame), approach (servo frame)} or None
        self.order = [self.root] + [j.child for j in self.joints]
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])
        self.by_child = dict((j.child, j) for j in self.joints)
        self.by_name = dict((j.name, j) for j in self.joints)

    def fk(self, q):
        """Link frames in the root frame at the DESIGN joint angles q (== MuJoCo qpos of the reference
        robot.xml); `joint_zero_offset_rad` is not applied (module docstring, Q14)."""
        T = {self.root: np.eye(4)}
        for j, qi in zip(self.joints, q):
            T[j.child] = T[j.parent] @ j.origin @ rot_axis(j.axis, qi)
        return T

    def occurrence(self, servo_id):
        """(link name, servo frame in the link frame) of an occurrence, or (None, None)."""
        for name, link in self.links.items():
            for sid, T in link.occurrences:
                if sid == servo_id:
                    return name, T
        return None, None

    def all_occurrences(self):
        out = []
        for name in self.order:
            for sid, T in self.links[name].occurrences:
                out.append((sid, name, T))
        return out

    def gripper_occurrence(self):
        if not self.tool:
            return None, None
        return self.occurrence(self.tool["servo_id"])

    def tcp(self, q, T=None):
        T = T or self.fk(q)
        lname, Tocc = self.gripper_occurrence()
        Tw = T[lname] @ Tocc
        return apply(Tw, self.tool["point"]), Tw[:3, :3] @ np.asarray(self.tool["approach"], float)

    def servo_world(self, servo_id, T):
        """(servo frame in the root frame, carrier link name) at the configuration behind T."""
        name, Tocc = self.occurrence(servo_id)
        if name is None:
            return None, None
        return T[name] @ Tocc, name

    def axis_line_world(self, joint, T):
        """(point, direction) of a joint axis in the root frame."""
        Tc = T[joint.child]
        return Tc[:3, 3], Tc[:3, :3] @ joint.axis

    def horn_side_link(self, joint):
        """The link that mates onto the driving servo's horn: the (parent, child) member that
        does NOT carry the servo (kinematic_sketch.json `servo_placement_freedom`)."""
        carrier, _ = self.occurrence(joint.driven_by[0])
        if carrier == joint.parent:
            return joint.child
        if carrier == joint.child:
            return joint.parent
        raise ValueError("joint {}: driving servo {} is carried by {}, neither parent nor child".format(
            joint.name, joint.driven_by[0], carrier))
