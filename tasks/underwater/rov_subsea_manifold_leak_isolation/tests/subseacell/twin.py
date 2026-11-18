"""The two sensor-side copies of a site world (cell side; DESIGN v1.2 sections 7 and 13, gate G5 proposal 4, G6
finding "one ping 14 ms"; package P08).

The world is THREE models (subseacell.world builds the first and both of these from it):
  physics model   collision geometry only (+ the roof grating bars, which collide with nothing but are seen by the DVL);
                  stepped every 2 ms; the firmware's DVL / depth / AHRS run on it.
  RenderTwin      what the camera sees: a copy of the physics MjSpec with the vehicle's appearance meshes grafted on
                  (the digest's visual meshes, the same body frames), the physics-only proxies hidden, the tag plates as
                  ONE textured quad each (gen/tag_textures.py, carried by the site package), the two-colour handle paint,
                  fouling, paint marks and the stands' lifting eyes.  Never stepped: before a frame its qpos is set
                  from the physics qpos (identical layout, identical body frames -> bit-identical body poses) and only
                  the kinematics run.
  AcousticModel   what the scanning sonar's rays see: structure-level geometry only (sea bed, mudmats, frame, roof bars,
                  pipes, flanges, panel face / frame / rail, one cylinder per port receptacle instead of its 24 wall
                  staves, the cage; amendment 02: the static scenery, rocks as meshes - subseacell.scenery) plus mocap
                  copies of the moving bodies (valve handles, marker, stab, stand); the
                  vehicle is not in it (decision for uwkit 0.3.0's acoustic_self = "none": its self-returns all lie in
                  the transducer's 0.5 m ring-down zone).  Before a ping the mocap poses and the sonar head are copied
                  from the physics data.

Geom groups (render twin): 0 scene, 1 vehicle parts the camera does not see (its own housing), 2 vehicle parts it sees
(jaws, front), 3 physics-only proxies (hidden), 4 fouling (shown by the camera, ignored by the "imaged" predicate's
rays), 5 unused.

    tw = RenderTwin(phys_spec, phys_model, scene)       # scene = world.twin_scene() (what to paint where)
    tw.sync(phys_data); cam.capture(tw.data, t)         # cam = sensing.PlumeCamera(tw.model, ...)
    ac = AcousticModel(phys_model, scene); ac.sync(phys_data); sonar.ping_now(t, ac.data)
"""
from __future__ import annotations

import functools
import math
import os
from typing import Optional

import numpy as np
import mujoco

GROUP_SCENE, GROUP_VIS_HIDDEN, GROUP_VIS_CAMERA, GROUP_PROXY, GROUP_FOUL, GROUP_HIDDEN = 0, 1, 2, 3, 4, 5
VEHICLE_BODIES = ("hull", "cam_tilt", "jaw_upper", "jaw_lower", "sonar", "dvl")
N_TEXROLE = 10
PAINT_ORANGE = (1.00, 0.45, 0.05, 1.0)          # panel standard two-colour handle paint (G4 panel_std)
PAINT_WHITE = (0.92, 0.92, 0.90, 1.0)
PLATE_RGBA = (230 / 255.0, 230 / 255.0, 224 / 255.0, 1.0)
EYE_RGBA = (1.0, 0.45, 0.05, 1.0)
LAYDOWN_RGBA = (0.95, 0.95, 0.95, 1.0)
QUAD_OFFSET_M = 0.0005                          # textured quad in front of its plate box (no z-fighting)
VISUAL_MESHES_ENV = "SUBSEACELL_VISUAL_MESHES"


def _mat_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.asarray(R, float)).reshape(-1))
    return q


def _pad_textures(spec):
    """MjSpec.copy() corrupts memory (MuJoCo 3.14.0) on material texture lists shorter than mjNTEXROLE (UWKIT_VIZ Q1)."""
    from uwkit.models.vehicle import pad_material_textures
    pad_material_textures(spec)


# ==================================================================================================== vehicle look
@functools.lru_cache(maxsize=2)
def vehicle_visuals(mesh_root: Optional[str] = None) -> dict:
    """The digest vehicle's appearance (groups 1 and 2) as the kit's builder places it: meshes (file, scale, ...),
    materials and geoms per body, plus the vehicle body frames (to check they match the physics build).  Built from a
    visual spec of the same configuration (mission, sea ballast, collision mesh, jaw shell); the spec is not compiled."""
    import uwkit
    from uwkit import refdata
    from . import vehicle as VH
    root = mesh_root or os.environ.get(VISUAL_MESHES_ENV) or refdata.mesh_root()
    vb = uwkit.build_vehicle(visual=True, mesh_dir=root, **VH.BUILD)
    s = vb.spec
    meshes, mats, geoms = {}, {}, []
    used = set()
    for bname in VEHICLE_BODIES:
        b = s.body(bname)
        for g in b.geoms:
            if g.group not in (GROUP_VIS_HIDDEN, GROUP_VIS_CAMERA):
                continue
            rec = dict(body=bname, name=g.name, type=int(g.type), size=[float(x) for x in g.size],
                       pos=[float(x) for x in g.pos], quat=[float(x) for x in g.quat], group=int(g.group),
                       rgba=[float(x) for x in g.rgba], material=g.material or "",
                       mesh=g.meshname if g.type == mujoco.mjtGeom.mjGEOM_MESH else "")
            geoms.append(rec)
            if rec["mesh"]:
                used.add(rec["mesh"])
            if rec["material"]:
                mats.setdefault(rec["material"], None)
    for me in s.meshes:
        if me.name in used:
            f = me.file if os.path.isabs(me.file) else os.path.join(s.meshdir or "", me.file)
            meshes[me.name] = dict(file=f, scale=[float(x) for x in me.scale], refpos=[float(x) for x in me.refpos],
                                   refquat=[float(x) for x in me.refquat], smoothnormal=bool(me.smoothnormal))
    for mt in s.materials:
        if mt.name in mats:
            mats[mt.name] = dict(rgba=[float(x) for x in mt.rgba], specular=float(mt.specular),
                                 shininess=float(mt.shininess), emission=float(mt.emission),
                                 reflectance=float(mt.reflectance))
    bodies = {n: dict(pos=[float(x) for x in s.body(n).pos], quat=[float(x) for x in s.body(n).quat])
              for n in VEHICLE_BODIES}
    return dict(root=root, meshes=meshes, materials=mats, geoms=geoms, bodies=bodies)


def visual_mesh_files(mesh_root: Optional[str] = None) -> list:
    """Absolute paths of the appearance meshes the render twin loads (for pinning them next to the collision set)."""
    return sorted({v["file"] for v in vehicle_visuals(mesh_root)["meshes"].values()})


class PhysicsPose:
    """Forward kinematics of the physics model at the physics data's CURRENT qpos, on a private MjData (the physics
    data's own xpos / site_xpos are those of the start of the last step: mj_step integrates qpos after computing them).
    The physics data is never touched, so taking frames or pings cannot change the trajectory."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)

    def __call__(self, phys_data: mujoco.MjData) -> mujoco.MjData:
        d = self.d
        d.qpos[:] = phys_data.qpos
        if self.m.nmocap:
            d.mocap_pos[:] = phys_data.mocap_pos
            d.mocap_quat[:] = phys_data.mocap_quat
        mujoco.mj_kinematics(self.m, d)
        return d


# ==================================================================================================== render twin
def _quad_mesh(spec, name, w, h):
    """Flat textured quad w x h in its local XY plane, facing +Z; UV (0, 0) = top-left (texture row 0)."""
    me = spec.add_mesh(name=name)
    me.uservert = [-w / 2, -h / 2, 0.0, w / 2, -h / 2, 0.0, w / 2, h / 2, 0.0, -w / 2, h / 2, 0.0]
    me.userface = [0, 1, 2, 0, 2, 3]
    me.usertexcoord = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    return me


def _add_texture(spec, name, img):
    img = np.ascontiguousarray(img, np.uint8)
    t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=int(img.shape[1]),
                         height=int(img.shape[0]), nchannel=3)
    t.data = img.tobytes()
    tex = [""] * N_TEXROLE
    tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = name
    spec.add_material(name=f"mat_{name}", textures=tex, rgba=[1.0, 1.0, 1.0, 1.0], specular=0.0, shininess=0.0)
    return f"mat_{name}"


class RenderTwin:
    """Render-only copy of the physics world.  scene (from SiteWorld.twin_scene()):
        plates     [{name, centre_w, R_w (X right, Y up, Z out of the face), size, thickness, image, fouling}]
        paddles    [{body, geom, centre_b, R_bp, size (depth, length, thickness)}]  -> two-colour halves, collision box hidden
        hide       [geom names of the physics model to hide]   delete [geom names to drop]
        eyes       [{body, centre_b, axis_b (ring normal)}]   lifting-eye rings
        segments   [{name, a, b, radius, rgba}]                paint lines (laydown area)
        boxes      [{name, centre, half, axes, rgba}]          look-only primitives (zone paint ...)
        mesh_root  appearance mesh library (None: env SUBSEACELL_VISUAL_MESHES, else the digest's mesh root)"""

    def __init__(self, phys_spec: mujoco.MjSpec, phys_model: mujoco.MjModel, scene: dict):
        _pad_textures(phys_spec)
        s = phys_spec.copy()
        self.scene = scene
        vis = vehicle_visuals(scene.get("mesh_root"))
        for n, fr in vis["bodies"].items():
            if n == VEHICLE_BODIES[0]:
                continue                                         # the root's pose is the free joint's qpos
            b = {x.name: x for x in s.bodies}.get(n)
            if b is None or not np.allclose(b.pos, fr["pos"], atol=1e-12) or not np.allclose(b.quat, fr["quat"], atol=1e-12):
                raise RuntimeError(f"vehicle body {n}: frame of the appearance build differs from the physics build")
        # MuJoCo 3.14: after MjSpec.delete() the name lookups (spec.geom(name), spec.body(name)) can return the wrong
        # element (measured: a copied spec with deleted geoms handed back newly added geoms for old names).  So every
        # element is looked up by iteration first, and all deletions happen last, right before compile.
        bodies = {b.name: b for b in s.bodies}
        geoms = {g.name: g for g in s.geoms}
        for n in scene.get("hide", ()):
            geoms[n].group = GROUP_PROXY
        # ---------------------------------------------------------------- vehicle appearance
        for mname, mt in vis["materials"].items():
            tex = [""] * N_TEXROLE
            s.add_material(name=mname, textures=tex, rgba=mt["rgba"], specular=mt["specular"], shininess=mt["shininess"],
                           emission=mt["emission"], reflectance=mt["reflectance"])
        for mname, me in vis["meshes"].items():
            m_ = s.add_mesh(name=f"vis_{mname}", file=me["file"])
            m_.scale, m_.refpos, m_.refquat = me["scale"], me["refpos"], me["refquat"]
            m_.smoothnormal = me["smoothnormal"]
        for g in vis["geoms"]:
            kw = dict(name=f"vis_{g['name']}", type=mujoco.mjtGeom(g["type"]), pos=g["pos"], quat=g["quat"],
                      group=g["group"], rgba=g["rgba"], contype=0, conaffinity=0, mass=0.0)
            if g["mesh"]:
                kw["meshname"] = f"vis_{g['mesh']}"
            else:
                kw["size"] = g["size"]
            if g["material"]:
                kw["material"] = g["material"]
            bodies[g["body"]].add_geom(**kw)
        self.n_vehicle_visuals = len(vis["geoms"])
        # ---------------------------------------------------------------- tag plates (one textured quad each)
        wb = s.worldbody
        quads, self.plates = {}, {}
        for k, pl in enumerate(scene.get("plates", ())):
            w, h = pl["size"]
            th = float(pl["thickness"])
            key = f"tagquad_{int(round(w * 1e4))}x{int(round(h * 1e4))}"
            if key not in quads:
                _quad_mesh(s, key, w, h)
                quads[key] = True
            R = np.asarray(pl["R_w"], float)
            q = _mat_quat(R)
            c = np.asarray(pl["centre_w"], float)                 # plate centre on the face plane
            wb.add_geom(name=f"plate_{pl['name']}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[w / 2, h / 2, th / 2],
                        pos=list(c + R[:, 2] * th / 2), quat=list(q), rgba=list(PLATE_RGBA), group=GROUP_SCENE,
                        contype=0, conaffinity=0)
            mat = _add_texture(s, f"tag_{k:02d}", pl["image"])
            wb.add_geom(name=f"plateface_{pl['name']}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=key, material=mat,
                        pos=list(c + R[:, 2] * (th + QUAD_OFFSET_M)), quat=list(q), group=GROUP_SCENE, contype=0,
                        conaffinity=0)
            for j, fp in enumerate(pl.get("fouling") or ()):
                P = c + R @ (np.asarray(fp["centre"], float) + np.array([0.0, 0.0, th]))
                qf = _mat_quat(R @ _quat_mat(fp["quat"]))
                gt = mujoco.mjtGeom.mjGEOM_BOX if fp["kind"] == "box" else mujoco.mjtGeom.mjGEOM_ELLIPSOID
                wb.add_geom(name=f"foul_{pl['name']}_{j}", type=gt, size=list(fp["size"]), pos=list(P), quat=list(qf),
                            rgba=list(fp["rgba"]), group=GROUP_FOUL, contype=0, conaffinity=0)
            self.plates[pl["name"]] = dict(centre_w=c, R_w=R, size=(w, h), front_z=th, code_id=pl.get("code_id"),
                                           text=pl.get("text"), fouled=bool(pl.get("fouling")))
        # ---------------------------------------------------------------- handle paint
        for pd in scene.get("paddles", ()):
            b = bodies[pd["body"]]
            geoms[pd["geom"]].group = GROUP_PROXY
            D, L, T = pd["size"]
            Rb = np.asarray(pd["R_bp"], float)
            c = np.asarray(pd["centre_b"], float)
            for half, sgn, col in (("orange", -1.0, PAINT_ORANGE), ("white", 1.0, PAINT_WHITE)):
                b.add_geom(name=f"{pd['body']}_paint_{half}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[D / 2, L / 4, T / 2],
                           pos=list(c + Rb @ np.array([0.0, sgn * L / 4, 0.0])), quat=list(_mat_quat(Rb)),
                           rgba=list(col), group=GROUP_SCENE, contype=0, conaffinity=0, mass=0.0)
        # ---------------------------------------------------------------- lifting eyes, paint marks, look-only boxes
        for e in scene.get("eyes", ()):
            b = bodies[e["body"]]
            c = np.asarray(e["centre_b"], float)
            rr = float(e["ring_r"])
            u, v = np.asarray(e["u_b"], float), np.asarray(e["v_b"], float)      # the ring's plane
            for k in range(12):
                a0, a1 = 2 * math.pi * k / 12, 2 * math.pi * (k + 1) / 12
                p0 = c + rr * (math.cos(a0) * u + math.sin(a0) * v)
                p1 = c + rr * (math.cos(a1) * u + math.sin(a1) * v)
                b.add_geom(name=f"{e['name']}_ring{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0035, 0, 0],
                           fromto=list(p0) + list(p1), rgba=list(EYE_RGBA), group=GROUP_SCENE, contype=0,
                           conaffinity=0, mass=0.0)
            if e.get("post") is not None:
                b.add_geom(name=f"{e['name']}_post", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0],
                           fromto=list(e["post"][0]) + list(e["post"][1]), rgba=list(EYE_RGBA), group=GROUP_SCENE,
                           contype=0, conaffinity=0, mass=0.0)
        for sg in scene.get("segments", ()):
            wb.add_geom(name=sg["name"], type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[sg["radius"], 0, 0],
                        fromto=list(sg["a"]) + list(sg["b"]), rgba=list(sg["rgba"]), group=GROUP_SCENE, contype=0,
                        conaffinity=0)
        for bx in scene.get("boxes", ()):
            wb.add_geom(name=bx["name"], type=mujoco.mjtGeom.mjGEOM_BOX, size=list(bx["half"]), pos=list(bx["centre"]),
                        quat=list(_mat_quat(bx["R"])), rgba=list(bx["rgba"]), group=GROUP_SCENE, contype=0,
                        conaffinity=0)
        # ---------------------------------------------------------------- deletions (last): vehicle collision geoms,
        # physics-only proxies, meshes nothing uses any more
        drop = set(scene.get("delete", ()))
        dead = [g for n in VEHICLE_BODIES for g in bodies[n].geoms if g.group == GROUP_PROXY]
        dead += [g for g in s.worldbody.geoms if g.name in drop]
        for g in dead:
            s.delete(g)
        used_meshes = {g.meshname for g in s.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
        for me in list(s.meshes):
            if me.name not in used_meshes:
                s.delete(me)
        # ---------------------------------------------------------------- compile (kinematics only)
        s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        vg = s.visual.global_
        vg.offwidth, vg.offheight = max(int(vg.offwidth), 640), max(int(vg.offheight), 480)
        self.spec = s
        self.model = s.compile()
        self.data = mujoco.MjData(self.model)
        m, pm = self.model, phys_model
        if m.nq != pm.nq or m.nbody != pm.nbody:
            raise RuntimeError("render twin: qpos layout or body tree differs from the physics model")
        for b in range(pm.nbody):
            if m.body(b).name != pm.body(b).name:
                raise RuntimeError("render twin: body order differs from the physics model")
        self.nmocap = int(m.nmocap)
        self.n_synced = 0
        self._phys = PhysicsPose(pm)
        mujoco.mj_kinematics(m, self.data)

    def sync(self, phys_data: mujoco.MjData):
        """Pose the twin like the physics state (qpos copy, kinematics only)."""
        d = self.data
        d.qpos[:] = phys_data.qpos
        d.time = phys_data.time
        if self.nmocap:
            d.mocap_pos[:] = phys_data.mocap_pos
            d.mocap_quat[:] = phys_data.mocap_quat
        mujoco.mj_kinematics(self.model, d)
        self.n_synced += 1

    def pose_error(self, phys_data: mujoco.MjData) -> dict:
        """Largest difference of every rigid body's position [m], quaternion and rotation-matrix entry between the twin
        (after sync) and the physics model's kinematics of the same state."""
        d, k = self.data, self._phys(phys_data)
        return dict(pos_m=float(np.abs(d.xpos - k.xpos).max()), quat=float(np.abs(d.xquat - k.xquat).max()),
                    xmat=float(np.abs(d.xmat - k.xmat).max()), n_bodies=int(d.xpos.shape[0]))

    def stats(self) -> dict:
        m = self.model
        return dict(ngeom=int(m.ngeom), nmesh=int(m.nmesh), ntex=int(m.ntex), nmat=int(m.nmat),
                    vehicle_visuals=self.n_vehicle_visuals, plates=len(self.plates),
                    groups=np.bincount(m.geom_group, minlength=6).tolist())


def _quat_mat(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


# ==================================================================================================== acoustic model
def _static_frame_ok(pm: mujoco.MjModel, bid: int) -> bool:
    """A body whose geom poses are world poses: the world body, or a joint-free child of it at the identity frame
    (the scenery's zone bodies, subseacell.scenery)."""
    if bid == 0:
        return True
    return (int(pm.body_parentid[bid]) == 0 and int(pm.body_weldid[bid]) == 0 and int(pm.body_jntnum[bid]) == 0
            and all(float(x) == 0.0 for x in pm.body_pos[bid])
            and [float(x) for x in pm.body_quat[bid]] == [1.0, 0.0, 0.0, 0.0])


class AcousticModel:
    """Structure-level geometry for the scanning sonar.  scene (from SiteWorld.acoustic_scene()):
        static_geoms  [geom names of the physics model's static geometry to copy (type, size, pose): geoms of the world
                       body or of a joint-free child of it at the identity frame (the scenery's zone bodies)]
        static_meshes [{name, mesh, file, scale, pos, quat}]  static mesh geoms (amendment 02: the scenery's rocks),
                       rebuilt from the same OBJ at the same scale and spec pose as in the physics spec, so that MuJoCo's
                       mesh re-centring gives the same compiled geom pose and vertices in both models
        cylinders     [{name, a, b, radius, acls}]          replacements (one per port receptacle)
        moving        [{body}]                               physics bodies mirrored as mocap bodies (their colliding geoms)
        classes       {geom name: acoustic class}            explicit classes (else uwkit's name rules)
        sonar_site    the physics site of the sonar head ("sonar_head")
    Built before the world's reset (subseacell.world.SiteWorld.__init__): the reset puts the dropped-object stand to
    sleep (contacts off), and only colliding geoms of a moving body are mirrored."""

    def __init__(self, phys_model: mujoco.MjModel, scene: dict):
        pm = phys_model
        s = mujoco.MjSpec()
        s.compiler.degree = False
        s.option.gravity = [0.0, 0.0, 0.0]
        s.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        wb = s.worldbody
        classes = dict(scene.get("classes", {}))
        self.static_names = []
        for n in scene.get("static_geoms", ()):
            g = pm.geom(n)
            gid = g.id
            gt = mujoco.mjtGeom(int(pm.geom_type[gid]))
            if not _static_frame_ok(pm, int(pm.geom_bodyid[gid])):
                raise ValueError(f"acoustic model: static geom {n} is not on a static body at the identity frame")
            kw = dict(name=n, type=gt, size=[float(x) for x in pm.geom_size[gid]], pos=[float(x) for x in pm.geom_pos[gid]],
                      quat=[float(x) for x in pm.geom_quat[gid]], group=0, contype=0, conaffinity=0)
            if gt == mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f"acoustic model: static mesh geom {n} must come as a static_meshes entry")
            if gt == mujoco.mjtGeom.mjGEOM_PLANE:
                kw["size"] = [max(kw["size"][0], 60.0), max(kw["size"][1], 60.0), 0.1]
            wb.add_geom(**kw)
            self.static_names.append(n)
        self.n_static_meshes = 0
        have = set()
        for e in scene.get("static_meshes", ()):
            if e["mesh"] not in have:
                me = s.add_mesh(name=e["mesh"], file=e["file"])
                me.scale = [float(e["scale"])] * 3
                have.add(e["mesh"])
            wb.add_geom(name=e["name"], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=e["mesh"], pos=list(e["pos"]),
                        quat=list(e["quat"]), group=0, contype=0, conaffinity=0)
            self.static_names.append(e["name"])
            self.n_static_meshes += 1
        for c in scene.get("cylinders", ()):
            wb.add_geom(name=c["name"], type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[c["radius"], 0, 0],
                        fromto=list(c["a"]) + list(c["b"]), group=0, contype=0, conaffinity=0)
            if c.get("acls"):
                classes[c["name"]] = c["acls"]
        # moving bodies -> mocap copies of their colliding geoms
        self.moving = []
        for mv in scene.get("moving", ()):
            bid = pm.body(mv["body"]).id
            b = wb.add_body(name=f"ac_{mv['body']}", mocap=True, pos=[0.0, 0.0, 0.0])
            n = 0
            for gid in range(pm.ngeom):
                if int(pm.geom_bodyid[gid]) != bid:
                    continue
                if int(pm.geom_contype[gid]) == 0 and int(pm.geom_conaffinity[gid]) == 0:
                    continue
                gt = mujoco.mjtGeom(int(pm.geom_type[gid]))
                if gt == mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                gname = "ac_" + (pm.geom(gid).name or f"{mv['body']}_{gid}")
                b.add_geom(name=gname, type=gt, size=[float(x) for x in pm.geom_size[gid]],
                           pos=[float(x) for x in pm.geom_pos[gid]], quat=[float(x) for x in pm.geom_quat[gid]], group=0,
                           contype=0, conaffinity=0)
                if mv.get("acls"):
                    classes[gname] = mv["acls"]
                n += 1
            self.moving.append((mv["body"], bid, n))
        # the sonar head (the vehicle itself is not in the model)
        sid = pm.site(scene.get("sonar_site", "sonar_head")).id
        self.phys_sid = sid
        mb = wb.add_body(name="ac_sonar_mount", mocap=True, pos=[0.0, 0.0, 0.0])
        mb.add_site(name=scene.get("sonar_site", "sonar_head"), pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
        self.spec = s
        self.model = s.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.classes = {k: v for k, v in classes.items() if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, k) >= 0}
        self.mount_id = m.body("ac_sonar_mount").id
        self.mount_mocap = int(m.body_mocapid[self.mount_id])
        self._mocap = np.array([int(m.body_mocapid[m.body(f"ac_{b}").id]) for b, _, _ in self.moving], np.int64)
        self._phys_bid = np.array([bid for _, bid, _ in self.moving], np.int64)
        self._ac_bid = np.array([m.body(f"ac_{b}").id for b, _, _ in self.moving], np.int64)
        self.n_synced = 0
        self._phys = PhysicsPose(pm)
        mujoco.mj_kinematics(m, self.data)

    def materials(self):
        from uwkit.sensors.materials import MaterialMap
        return MaterialMap(self.model, overrides=self.classes)

    def sync(self, phys_data: mujoco.MjData):
        """Pose the mocap copies and the sonar head like the physics state (its current qpos)."""
        d, k = self.data, self._phys(phys_data)
        if self._mocap.size:
            d.mocap_pos[self._mocap] = k.xpos[self._phys_bid]
            d.mocap_quat[self._mocap] = k.xquat[self._phys_bid]
        d.mocap_pos[self.mount_mocap] = k.site_xpos[self.phys_sid]
        d.mocap_quat[self.mount_mocap] = _mat_quat(k.site_xmat[self.phys_sid].reshape(3, 3))
        d.time = phys_data.time
        mujoco.mj_kinematics(self.model, d)
        self.n_synced += 1

    def pose_error(self, phys_data: mujoco.MjData) -> dict:
        d, k = self.data, self._phys(phys_data)
        dp = float(np.abs(d.xpos[self._ac_bid] - k.xpos[self._phys_bid]).max()) if self._ac_bid.size else 0.0
        dq = float(np.abs(d.xquat[self._ac_bid] - k.xquat[self._phys_bid]).max()) if self._ac_bid.size else 0.0
        sp = float(np.abs(d.site_xpos[0] - k.site_xpos[self.phys_sid]).max())
        sm = float(np.abs(d.site_xmat[0] - k.site_xmat[self.phys_sid]).max())
        return dict(pos_m=max(dp, sp), quat=dq, site_xmat=sm, n_bodies=int(self._ac_bid.size))

    def stats(self) -> dict:
        m = self.model
        out = dict(ngeom=int(m.ngeom), nbody=int(m.nbody), static=len(self.static_names),
                   moving=[(b, n) for b, _, n in self.moving])
        if self.n_static_meshes:
            out["static_meshes"] = self.n_static_meshes
        return out
