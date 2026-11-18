"""Visual twin of a vehicle build: a render-only copy of the physics MjSpec with animated parts.

    twin = VisualTwin(vb)               # vb = the VehicleBuild the simulator was compiled from (world included)
    twin.set_physics_qpos(qpos, joints) # pose of hull / camera tilt / jaws / free objects, copied by joint name
    twin.set_rotor_angles(phi)          # (n_thrusters,) rad about each site's +x axis
    twin.set_lamps([0, 0, 1, 1])        # 0..1 per lamp (light_1 .. light_4)
    twin.forward()                      # kinematics only (no dynamics, no collision)

What changes with respect to the physics model (the physics model itself is never touched; vb.spec is copied):
  - vehicle collision geoms (group 3) are dropped;
  - every thruster's single visual mesh is replaced by its stator mesh (nozzle with tail cone, stator base, nose cone,
    screws) at the same pose, plus a child body 'viz_rotor_<site>' on a hinge about the site axis through the propeller
    centre carrying the rotor meshes (hub, blades, optional contrasting marker on one blade);
  - each lamp gets its own glow material (lens / LED parts), a spot light and optional translucent beam cones;
  - one body-fixed camera per thruster ('viz_cam_<site>') on the nose-cone side looking along -axis (the vendor's
    convention for naming CW / CCW propellers), for close-ups.
Mass, inertia and joints of the physics model are irrelevant here: the twin is only posed (mj_kinematics), never stepped.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import mujoco

from .. import refdata
from ..frames import mat_to_quat
from .rotors import RotorSet

GROUP_COLLISION = 3
ROTOR_RGBA = (0.42, 0.44, 0.48, 1.0)       # propeller: grey, lighter than the nozzle (thruster_black) for readability
MARKER_RGBA = (1.0, 0.78, 0.10, 1.0)       # visual aid on one blade (not a real paint scheme)
LAMP_GLOW_RGB = (1.0, 0.96, 0.85)


def _quat_close(a, b, tol=1e-5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return abs(abs(float(a @ b)) - 1.0) < tol


def cone_mesh(half_angle_rad, length, n=48, r0=0.012):
    """Open truncated cone (beam shell), axis +x, radius r0 at x = 0 (lamp lens), no caps (vertices, faces)."""
    R = r0 + length * math.tan(half_angle_rad)
    V = []
    for i in range(n):
        a = 2 * math.pi * i / n
        V.append([0.0, r0 * math.cos(a), r0 * math.sin(a)])
        V.append([length, R * math.cos(a), R * math.sin(a)])
    F = []
    for i in range(n):
        j = (i + 1) % n
        a0, a1, b0, b1 = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
        F.append([a0, b0, a1])
        F.append([a1, b0, b1])
    return np.asarray(V, float), np.asarray(F, int)


# MjSpec.copy() padding (MuJoCo 3.14.0): one implementation, shared with VehicleBuild.export()
from ..models.vehicle import N_TEXROLE, pad_material_textures  # noqa: E402,F401


def _camera_quat(look_dir, up_hint):
    """Quaternion of a MuJoCo camera frame that looks along look_dir (camera -z) with image up close to up_hint."""
    z = -np.asarray(look_dir, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up_hint, float)
    y = up - (up @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.array([1.0, 0.0, 0.0]) - z[0] * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return mat_to_quat(np.column_stack([x, y, z]))


class VisualTwin:
    def __init__(self, vb, rotors: Optional[RotorSet] = None, marker: bool = True, lamps: bool = True,
                 lamp_cones: bool = True, thruster_cameras: bool = True, mesh_dir: Optional[str] = None,
                 cam_distance: float = 0.30, cam_fovy_deg: float = 25.0, lamp_cutoff_deg: float = 55.0):
        meta = vb.meta
        ref = meta.get("reference") or {}
        if not ref:
            raise ValueError("the visual twin needs a reference-vehicle build (uwkit.build_vehicle with a digest config)")
        self.config = ref["config"]
        self.rotors = rotors or RotorSet(self.config, sites=list(meta["thrusters"]))
        root = ref.get("mesh_root") or refdata.mesh_root()
        vizdir = mesh_dir or os.path.join(root, "viz")
        # MjSpec.copy() corrupts memory (MuJoCo 3.14.0) when a material carries a texture list shorter than mjNTEXROLE
        # (e.g. textures=["", "sand"]); pad such lists with "" first - a no-op for the compiled model (tested bit-exact)
        pad_material_textures(vb.spec)
        s = vb.spec.copy()
        hull = s.body(meta["vehicle_body"])
        veh_geoms = hull.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        # ---------------------------------------------------------------- drop the vehicle collision geoms
        for g in [g for g in veh_geoms if g.group == GROUP_COLLISION]:
            s.delete(g)
        # ---------------------------------------------------------------- thrusters: stator swap + rotor bodies
        man = refdata.load_manifest()
        lo = {i["id"]: os.path.join(root, i["mesh"]["visual_lo"]) for i in man["items"]
              if i.get("category") == "thruster" and i.get("mesh")}
        mesh_file = {m.name: (m.file if os.path.isabs(m.file) else os.path.join(s.meshdir or "", m.file)) for m in s.meshes}
        hull_geoms = [g for g in hull.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]
        added = {}

        def mesh(name, path):
            if name not in added:
                s.add_mesh(name=name, file=path)
                added[name] = True
            return name

        s.add_material(name="viz_rotor", rgba=list(ROTOR_RGBA), specular=0.35, shininess=0.4)
        s.add_material(name="viz_marker", rgba=list(MARKER_RGBA), specular=0.3, shininess=0.3, emission=0.35)
        self.rotor_bodies, self.rotor_geoms, self.stator_geoms = [], [], []
        for r in self.rotors.rotors:
            P = self.rotors.prototype(r.prototype)
            files = {k: os.path.join(vizdir, v) for k, v in P["mesh_files"].items()}
            for f in files.values():
                if not os.path.exists(f):
                    raise FileNotFoundError(f"{f} missing - run uwkit/viz/build_rotor_split.py")
            want_file = os.path.abspath(lo[r.item])
            q = mat_to_quat(r.T_B[:3, :3])
            match = [g for g in hull_geoms if os.path.abspath(mesh_file.get(g.meshname, "")) == want_file
                     and np.allclose(g.pos, r.T_B[:3, 3], atol=2e-6) and _quat_close(g.quat, q)]
            if len(match) != 1:
                raise RuntimeError(f"thruster {r.site}: found {len(match)} visual geoms for {r.item}")
            named = (ref.get("thruster_geoms") or {}).get(r.site)      # 0.3.0 builds: site -> visual geom names
            if named and match[0].name not in named:
                raise RuntimeError(f"thruster {r.site}: the builder's geom map {named} disagrees with the mesh / pose match "
                                   f"{match[0].name}")
            g = match[0]
            g.meshname = mesh(f"viz_stator_{r.prototype}", files["stator"])
            g.name = f"viz_stator_{r.site}"
            self.stator_geoms.append(g.name)
            b = hull.add_body(name=f"viz_rotor_{r.site}", pos=list(r.centre_B))
            b.explicitinertial = True                  # render-only body; never simulated
            b.mass, b.inertia = 1e-3, [1e-7, 1e-7, 1e-7]
            b.add_joint(name=f"viz_rotor_{r.site}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(r.axis_B))
            parts = [("hub", "viz_rotor"), ("blades", "viz_rotor"), ("marker", "viz_marker" if marker else "viz_rotor")]
            names = []
            for part, mat in parts:
                nm = f"viz_rotor_{r.site}_{part}"
                b.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH,
                           meshname=mesh(f"viz_rotor_{r.prototype}_{part}", files[f"rotor_{part}"]),
                           pos=list(r.T_B[:3, 3] - r.centre_B), quat=list(q), material=mat, contype=0, conaffinity=0,
                           group=1, mass=0.0)
                names.append(nm)
            self.rotor_bodies.append(b.name)
            self.rotor_geoms.append(names)
            if thruster_cameras:
                up = np.array([0.0, 0.0, 1.0]) if abs(r.axis_B[2]) < 0.7 else np.array([1.0, 0.0, 0.0])
                hull.add_camera(name=f"viz_cam_{r.site}", pos=list(r.centre_B + cam_distance * r.axis_B),
                                quat=list(_camera_quat(-r.axis_B, up)), fovy=cam_fovy_deg)
        # ---------------------------------------------------------------- lamps
        self.lamp_mats, self.lamp_lights, self.lamp_cones = [], [], []
        lamp_sites = sorted(x.name for x in s.sites if x.name.startswith("light_"))
        site_pos = {x.name: np.asarray(x.pos, float) for x in s.sites}
        lens = [g for g in hull.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.material == "rov_lens_glass"]
        if lamps and lamp_sites:
            base = s.material("rov_lens_glass")
            for k, sn in enumerate(lamp_sites):
                p = site_pos[sn]
                mine = [g for g in lens if np.linalg.norm(np.asarray(g.pos) - p) < 0.06]
                mat = s.add_material(name=f"viz_lamp_{k + 1}", rgba=list(base.rgba), specular=float(base.specular),
                                     shininess=float(base.shininess))
                for g in mine:
                    g.material = mat.name
                self.lamp_mats.append(mat.name)
                hull.add_light(name=f"viz_lamp_{k + 1}", pos=list(p + np.array([0.004, 0, 0])), dir=[1.0, 0.0, 0.0],
                               type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=lamp_cutoff_deg, exponent=6.0,
                               diffuse=[0.0, 0.0, 0.0], specular=[0.0, 0.0, 0.0], ambient=[0.0, 0.0, 0.0],
                               attenuation=[1.0, 0.0, 0.08], castshadow=False, active=False)
                self.lamp_lights.append(f"viz_lamp_{k + 1}")
                cones = []
                if lamp_cones:
                    for j, (L, ang) in enumerate(((0.35, 0.36), (0.7, 0.34), (1.2, 0.32))):
                        mname = f"viz_cone_{j}"
                        if mname not in added:
                            Vc, Fc = cone_mesh(ang, L)
                            mm = s.add_mesh(name=mname)
                            mm.uservert = Vc.reshape(-1).tolist()
                            mm.userface = Fc.reshape(-1).tolist()
                            mm.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                            added[mname] = True
                        gn = f"viz_lamp_{k + 1}_cone_{j}"
                        hull.add_geom(name=gn, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                                      pos=list(p + np.array([0.003, 0, 0])), rgba=[1.0, 0.97, 0.85, 0.0],
                                      contype=0, conaffinity=0, group=1, mass=0.0)
                        cones.append(gn)
                self.lamp_cones.append(cones)
        self.lamp_sites = lamp_sites
        # ---------------------------------------------------------------- compile
        s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.spec = s
        self.model = s.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hull_id = m.body(meta["vehicle_body"]).id
        self.rotor_body_ids = np.array([m.body(n).id for n in self.rotor_bodies])
        self.rotor_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.rotor_bodies])
        self.rotor_geom_ids = [[m.geom(n).id for n in names] for names in self.rotor_geoms]
        self.stator_geom_ids = [m.geom(n).id for n in self.stator_geoms]
        self.lamp_mat_ids = [m.material(n).id for n in self.lamp_mats]
        self.lamp_light_ids = [m.light(n).id for n in self.lamp_lights]
        self.lamp_cone_ids = [[m.geom(n).id for n in c] for c in self.lamp_cones]
        self._lamp_base = [(m.mat_rgba[i].copy(), float(m.mat_emission[i])) for i in self.lamp_mat_ids]
        self._cone_alpha = (0.045, 0.03, 0.02)
        self.vehicle_geom_ids = np.array([g for g in range(m.ngeom) if self._in_vehicle(m.geom_bodyid[g])])
        self.cameras = {r.site: m.camera(f"viz_cam_{r.site}").id for r in self.rotors.rotors} if thruster_cameras else {}
        from .log import joint_table
        self.joints = joint_table(m)
        self._map_cache = {}
        self.lamp_levels = np.zeros(len(self.lamp_mats))
        mujoco.mj_kinematics(m, self.data)

    def _in_vehicle(self, b):
        m = self.model
        while b > 0:
            if b == self.hull_id:
                return True
            b = int(m.body_parentid[b])
        return False

    # ------------------------------------------------------------------ state
    def _qmap(self, phys_joints: dict):
        key = id(phys_joints)
        if key not in self._map_cache:
            src, dst = [], []
            for nm, (t, a, nq) in phys_joints.items():
                if nm in self.joints and self.joints[nm][0] == t:
                    src.append(np.arange(a, a + nq))
                    dst.append(np.arange(self.joints[nm][1], self.joints[nm][1] + nq))
            self._map_cache = {key: (np.concatenate(src), np.concatenate(dst))}
        return self._map_cache[key]

    def set_physics_qpos(self, qpos, phys_joints: dict):
        """Copy the physics qpos into the twin by joint name (phys_joints = SimLog.meta['joints'] or joint_table(m))."""
        src, dst = self._qmap(phys_joints)
        self.data.qpos[dst] = np.asarray(qpos, float)[src]

    def set_from_sim(self, sim):
        from .log import joint_table
        if not hasattr(self, "_sim_joints"):
            self._sim_joints = joint_table(sim.model)
        self.set_physics_qpos(sim.data.qpos, self._sim_joints)

    def set_rotor_angles(self, phi):
        self.data.qpos[self.rotor_qadr] = np.mod(np.asarray(phi, float), 2 * math.pi)

    def set_lamps(self, levels):
        m = self.model
        lv = np.clip(np.asarray(levels, float).reshape(-1), 0, 1)
        self.lamp_levels = lv
        for k in range(min(len(lv), len(self.lamp_mat_ids))):
            x = float(lv[k])
            i = self.lamp_mat_ids[k]
            rgba0, em0 = self._lamp_base[k]
            m.mat_emission[i] = em0 + (1.0 - em0) * x
            m.mat_rgba[i] = rgba0 * (1 - x) + np.array([*LAMP_GLOW_RGB, 1.0]) * x
            li = self.lamp_light_ids[k]
            m.light_active[li] = 1 if x > 0 else 0
            m.light_diffuse[li] = np.array([1.0, 0.95, 0.85]) * 0.9 * x
            m.light_specular[li] = np.array([0.3, 0.3, 0.3]) * x
            for j, g in enumerate(self.lamp_cone_ids[k] if k < len(self.lamp_cone_ids) else []):
                m.geom_rgba[g, 3] = self._cone_alpha[j] * x

    def forward(self):
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)

    # ------------------------------------------------------------------ geometry queries (after forward)
    def rotor_axes_world(self):
        R = self.data.xmat[self.hull_id].reshape(3, 3)
        return np.array([R @ r.axis_B for r in self.rotors.rotors])

    def rotor_centres_world(self):
        return self.data.xpos[self.rotor_body_ids].copy()

    def body_to_world(self, p_B):
        R = self.data.xmat[self.hull_id].reshape(3, 3)
        return self.data.xpos[self.hull_id] + np.asarray(p_B, float) @ R.T

    def hull_pose(self):
        return self.data.xpos[self.hull_id].copy(), self.data.xmat[self.hull_id].reshape(3, 3).copy()
