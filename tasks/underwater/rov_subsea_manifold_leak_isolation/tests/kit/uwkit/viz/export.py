"""Exchange format for other renderers (entry point for USD time samples, e.g. an RTX demo video): the static part
hierarchy of the visual twin + per-frame local transforms of every movable part.  This module writes plain NumPy / JSON;
it does not depend on any USD library.

    static = static_scene(twin)                                   # bodies (tree), geoms (mesh file, local pose, colour)
    anim = part_transforms(log, twin, fps=30, mode="limited")     # per frame, per movable part: local pos + quat
    save_animation("run_anim", static, anim)                      # -> run_anim.json (static + meta) + run_anim.npz

Format (schema 'uwkit.viz.anim/1'):
  static.bodies[i] = {name, parent, joint_type ('free' | 'hinge' | 'slide' | 'fixed'), rest_pos, rest_quat}
  static.geoms[j]  = {name, body, type, mesh_file | size, pos, quat (local to the body), rgba, material}
  anim.parts       = movable body names (hull and free objects: parent 'world'; camera tilt, jaws, rotors: parent hull)
  anim.times [F], anim.pos [F, P, 3], anim.quat [F, P, 4] (w, x, y, z) = transform of the part frame relative to its
  parent frame (USD: xformOp:translate / xformOp:orient time samples on the part prim, with the geoms as children at
  their static local poses); anim.lamps [F, n_lamps] (0..1 -> emissive / light intensity), anim.rotor_rpm [F, n].
Units: metres, seconds; world z up (MuJoCo) - a USD stage with upAxis = 'Z', metersPerUnit = 1 takes the data as is.
"""
from __future__ import annotations

import json

import numpy as np
import mujoco

from ..frames import mat_to_quat

SCHEMA = "uwkit.viz.anim/1"
_JT = {int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
       int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_BALL): "ball"}


def _body_joint_type(m, b):
    if m.body_jntnum[b] == 0:
        return "fixed"
    return _JT[int(m.jnt_type[m.body_jntadr[b]])]


def movable_bodies(twin):
    m = twin.model
    return [b for b in range(1, m.nbody) if m.body_jntnum[b] > 0]


def static_scene(twin) -> dict:
    m = twin.model
    mesh_files = {}
    for mm in twin.spec.meshes:
        if mm.file:
            mesh_files[mm.name] = mm.file
    bodies = [dict(name=m.body(b).name, parent=m.body(int(m.body_parentid[b])).name if b > 0 else None,
                   joint_type=_body_joint_type(m, b), rest_pos=m.body_pos[b].tolist(), rest_quat=m.body_quat[b].tolist())
              for b in range(m.nbody)]
    geoms = []
    for g in range(m.ngeom):
        t = int(m.geom_type[g])
        rec = dict(name=m.geom(g).name, body=m.body(int(m.geom_bodyid[g])).name,
                   type=mujoco.mjtGeom(t).name.replace("mjGEOM_", "").lower(), pos=m.geom_pos[g].tolist(),
                   quat=m.geom_quat[g].tolist(), size=m.geom_size[g].tolist(),
                   rgba=(m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]).tolist(),
                   material=m.material(int(m.geom_matid[g])).name if m.geom_matid[g] >= 0 else None,
                   group=int(m.geom_group[g]))
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            nm = m.mesh(int(m.geom_dataid[g])).name
            rec["mesh"] = nm
            rec["mesh_file"] = mesh_files.get(nm)
        geoms.append(rec)
    lamps = [dict(name=n, site=s) for n, s in zip(twin.lamp_lights, twin.lamp_sites)]
    return dict(schema=SCHEMA, units="m, s, quaternion (w, x, y, z)", up_axis="Z", bodies=bodies, geoms=geoms,
                lamps=lamps, rotors=[dict(site=r.site, body=f"viz_rotor_{r.site}", axis_B=r.axis_B.tolist(),
                                          centre_B=r.centre_B.tolist(), propeller=r.propeller,
                                          spin_sign_forward=r.spin_sign_forward) for r in twin.rotors.rotors])


def part_transforms(log, twin, fps=30.0, mode="limited", t_range=None, display=None) -> dict:
    """Local transforms (relative to the parent body) of every movable part at every video frame of the log."""
    from .render import AngleSource
    m, d = twin.model, twin.data
    rows = log.rows_for_fps(fps)
    if t_range is not None:
        rows = rows[(log.t[rows] >= t_range[0] - 1e-9) & (log.t[rows] <= t_range[1] + 1e-9)]
    src = AngleSource(log, twin.rotors.spin, fps, mode, display=display)
    parts = movable_bodies(twin)
    F, P = len(rows), len(parts)
    pos = np.zeros((F, P, 3))
    quat = np.zeros((F, P, 4))
    joints = log.meta["joints"]
    for i, row in enumerate(rows):
        twin.set_physics_qpos(log.qpos[row], joints)
        twin.set_rotor_angles(src.at(row).phi)
        mujoco.mj_kinematics(m, d)
        for j, b in enumerate(parts):
            pb = int(m.body_parentid[b])
            Rp = d.xmat[pb].reshape(3, 3)
            Rc = d.xmat[b].reshape(3, 3)
            pos[i, j] = Rp.T @ (d.xpos[b] - d.xpos[pb])
            quat[i, j] = mat_to_quat(Rp.T @ Rc)
    return dict(schema=SCHEMA, fps=float(fps), mode=mode, times=log.t[rows].copy(), rows=rows.copy(),
                parts=[m.body(b).name for b in parts], parents=[m.body(int(m.body_parentid[b])).name for b in parts],
                pos=pos, quat=quat, lamps=log.lights[rows].copy(), rotor_rpm=log.rpm[rows].copy(),
                labels=[log.labels_at(r) for r in rows])


def save_animation(prefix, static: dict, anim: dict):
    meta = dict(static=static, **{k: v for k, v in anim.items() if k in ("schema", "fps", "mode", "parts", "parents",
                                                                       "labels")})
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    np.savez_compressed(prefix + ".npz", **{k: anim[k] for k in ("times", "rows", "pos", "quat", "lamps", "rotor_rpm")})
    return prefix + ".json", prefix + ".npz"
