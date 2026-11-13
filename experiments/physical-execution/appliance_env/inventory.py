"""Inventory of RoboCasa's stove, stovetop, microwave and oven fixture models (P1 of
docs/dev/20261112-appliance-tending-task-design.md).

Loads every model.xml of the four kinds standalone in plain MuJoCo (no RoboCasa runtime), as the
asset archive ships it, and records what the task needs to know: whether it compiles, its
bounding box, its joints (kind, range, damping, friction, world anchor and axis), its burner
sites and button geoms, and the mass of each moving body. RoboCasa's frame convention: the
appliance front faces -y, z up.

Run (any Python with MuJoCo 3.16 and NumPy):
    python -I inventory.py --assets ~/arb-data/robocasa-assets/extracted/fixtures \
        --out outputs/inventory.json
"""

import argparse
import json
import os
import re
import sys

import mujoco
import numpy as np

KINDS = ("stoves", "stovetops", "microwaves", "ovens")
JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def model_bbox(m, d):
    """Axis-aligned box over all geoms (mesh geoms by their AABB), in the model frame."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_rgba[g, 3] == 0 and m.geom_contype[g] == 0:
            continue
        c = m.geom_aabb[g, :3]
        h = m.geom_aabb[g, 3:]
        R = d.geom_xmat[g].reshape(3, 3)
        center = d.geom_xpos[g] + R @ c
        half = np.abs(R) @ h
        lo = np.minimum(lo, center - half)
        hi = np.maximum(hi, center + half)
    return lo, hi


def inspect(path):
    rec = {"path": path}
    try:
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as exc:  # keep going: a model that does not compile is a finding
        rec["compiles"] = False
        rec["error"] = str(exc).splitlines()[0][:300]
        return rec
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rec["compiles"] = True
    lo, hi = model_bbox(m, d)
    rec["bbox_lo"] = np.round(lo, 4).tolist()
    rec["bbox_hi"] = np.round(hi, 4).tolist()
    rec["size"] = np.round(hi - lo, 4).tolist()
    joints = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        body = m.jnt_bodyid[j]
        dof = m.jnt_dofadr[j]
        joints.append({
            "name": name,
            "type": JOINT_TYPES.get(int(m.jnt_type[j]), str(m.jnt_type[j])),
            "range": np.round(m.jnt_range[j], 4).tolist() if m.jnt_limited[j] else None,
            "anchor": np.round(d.xanchor[j], 4).tolist(),
            "axis": np.round(d.xaxis[j], 4).tolist(),
            "damping": float(m.dof_damping[dof]),
            "frictionloss": float(m.dof_frictionloss[dof]),
            "armature": float(m.dof_armature[dof]),
            "stiffness": float(m.jnt_stiffness[j]),
            "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body),
            "subtree_mass": round(float(m.body_subtreemass[body]), 4),
        })
    rec["joints"] = joints
    rec["burner_sites"] = {}
    for s in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or ""
        mt = re.fullmatch(r"burner_on_(\w+)", name)
        if mt:
            rec["burner_sites"][mt.group(1)] = np.round(d.site_xpos[s], 4).tolist()
    rec["buttons"] = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "button" in name:
            rec["buttons"][name] = {
                "pos": np.round(d.geom_xpos[g], 4).tolist(),
                "size": np.round(m.geom_size[g], 4).tolist(),
                "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]),
            }
    rec["nmesh"] = int(m.nmesh)
    rec["ngeom"] = int(m.ngeom)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="the extracted fixtures/ folder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.path.expanduser(args.assets)
    result = {}
    for kind in KINDS:
        folder = os.path.join(root, kind)
        result[kind] = {}
        for name in sorted(os.listdir(folder)):
            xml = os.path.join(folder, name, "model.xml")
            if os.path.isfile(xml):
                result[kind][name] = inspect(xml)
                print(kind, name, "ok" if result[kind][name]["compiles"] else "FAIL", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
