"""Render twin of the hand assembly cell for MuJoCo's classic renderer (the agent cameras' fallback renderer and the
base of the Filament scene, agent_render.py).

The twin is the episode's physics MjSpec (same bodies and joint names: `session.sync_from(model, data)` copies the
state by joint name) plus visual-only additions: every unit's render meshes (the hidden kit's meshes/visual/: one mesh
per kind, link and material, the authors' solids and the servos' vendor solids tessellated from their B-reps; ONE mesh
asset per file, shared by the three finger modules), the bench frame, floor and backdrop walls, the three datum boards
(procedural checkerboards of `render.scale_refs`, their textures in sRGB: the bench board on the world, the fixture
board flat on the fixture's base plate, the hand plate on the Panda hand) and the episode's lights. Collision geoms
(group 3) stay in the spec; the renderer draws groups 0-2. One render session per episode (`open_session`).

Materials: one per material key of kit_info.json (`vis_<key>`: its sRGB colour; agent_render's look gives the
physically based terms per key). `omit` (render_omit.json, applied by agent_render only) leaves whole pieces out of a
mesh file by their face ranges; no triangle is changed.
"""

import math
import os

import numpy as np

from ..assembly_info.cell_render import attach_ref, board_ref

__all__ = ["scene", "open_session", "render_spec", "board_poses", "add_visuals", "BOARD_GEOMS", "triangle_count"]

_SPECS = {}
BOARD_GEOMS = ("bench_board", "fixture_board", "hand_plate")


def scene(key):
    """SceneSpec builder: a fresh copy of the registered render MjSpec."""
    return _SPECS[key].copy()


def _frame(z, x):
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - z * float(np.dot(x, z))
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    return T


def board_poses(server):
    """{name: (parent body, T_in_parent, ref)} of the three boards for this episode"""
    from ..assembly_info.cell_info import BENCH_BOARD, HAND_PLATE
    from . import fixture as FX
    out = {}
    b = BENCH_BOARD
    T = _frame([0, 0, 1], [math.cos(math.radians(b["yaw_deg"])), math.sin(math.radians(b["yaw_deg"])), 0])
    T[:3, 3] = b["pos"]
    out["bench_board"] = ("world", T, board_ref("bench_board", b["cols"], b["rows"], b["square_mm"]))
    fc = (server.cfg.get("cell") or {}).get("fixture")
    fb = FX.merged(fc)["board"]
    out["fixture_board"] = ("fixture", FX.board_pose_in_fixture(fc), board_ref("fixture_board", fb["cols"], fb["rows"], fb["square_mm"],
                                                                            fb["margin_mm"], fb["thickness_mm"]))
    h = HAND_PLATE
    T = _frame(h["normal_in_hand"], h["cols_along_in_hand"])
    T[:3, 3] = np.asarray(h["pos_in_hand_m"], float)
    J = np.asarray(server.episode["hand_plate_jitter"], float)
    out["hand_plate"] = ("panda/hand", T @ J, board_ref("hand_plate", h["cols"], h["rows"], h["square_mm"], h["margin_mm"], h["thickness_mm"]))
    return out


def _materials(spec, kit_info):
    """one material per material key of kit_info.json (its sRGB colour); returns {key: material name}"""
    made = {}
    have = set(mm.name for mm in spec.materials)
    for key, c in ((kit_info or {}).get("materials") or {}).items():
        name = "vis_" + key
        if name not in have:
            rgb = list(c.get("rgba_srgb", [0.5, 0.5, 0.5]))[:3]
            m = spec.add_material(name=name, rgba=rgb + [1.0])
            m.specular = 0.3 if float(c.get("metallic", 0.0)) < 0.5 else 0.8
            m.shininess = float(max(0.05, 1.0 - float(c.get("roughness", 0.6))))
        made[key] = name
    return made


def _keep_faces(F, drop):
    """F without the face ranges in `drop` ([(first, count), ...])"""
    if not drop:
        return F
    keep = np.ones(len(F), bool)
    for a, n in drop:
        keep[int(a):int(a) + int(n)] = False
    return F[keep]


def add_visuals(spec, server, omit=None, targets=None):
    """the units' render meshes on every unit body of the twin (visual only, group 2, no mass); one mesh asset per
    file. `omit` = {"files": {file: [(first face, count), ...]}}: those whole pieces are left out of the meshes (the
    fully hidden ones; no piece is changed). `targets` = [(unit, {link: body name})] (default: the cell's units)."""
    import mujoco
    from .kit import read_ply
    cell = server.cell
    kit = cell.kit
    gt = getattr(cell, "gt_dir", None) or server.kit_dir
    mats = _materials(spec, server.kit_info or {})
    meshes = {mm.name: mm for mm in spec.meshes if (mm.name or "").startswith("vismesh_")}
    if targets is None:
        targets = [(u, inf["links"]) for u, inf in cell.info.items()]
    drop = (omit or {}).get("files") or {}
    bodies = {b.name: b for b in spec.bodies}
    rec = {"geoms": 0, "triangles_drawn": 0, "triangles_omitted": 0}
    tri = {}
    for unit, links in targets:
        kind = kit.kind_of(unit)
        for v in kit.kinds[kind].get("visual", []):
            key = "vismesh_" + os.path.basename(v["file"])[:-4]
            if key not in meshes:
                V, F = read_ply(os.path.join(gt, v["file"]))
                F2 = _keep_faces(F, drop.get(v["file"]))
                tri[key] = (len(F2), len(F) - len(F2))
                if len(F2) == 0:
                    meshes[key] = None
                else:
                    ms = spec.add_mesh(name=key)
                    ms.uservert = (V * 1e-3).ravel().tolist()
                    ms.userface = F2.ravel().tolist()
                    ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                    meshes[key] = ms
            if key in tri:
                rec["triangles_omitted"] += tri[key][1]
            if meshes[key] is None:
                continue
            mat = v["material"]
            kw = {"material": mats[mat]} if mat in mats else {"rgba": [0.5, 0.5, 0.5, 1.0]}
            bodies[links[v["link"]]].add_geom(name="%s__v_%s_%s" % (unit, v["link"], mat), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=key,
                                              contype=0, conaffinity=0, group=2, mass=0.0, **kw)
            rec["geoms"] += 1
            rec["triangles_drawn"] += tri[key][0] if key in tri else len(meshes[key].userface) // 3
    server.render_twin_record = rec
    return meshes


def render_spec(server):
    """The episode's render MjSpec (a copy of the physics spec + the visual meshes + visual additions)."""
    import mujoco
    spec = server.cell.spec.copy()
    add_visuals(spec, server, omit=getattr(server, "render_omit", None))
    w = spec.worldbody
    bc = server.cell.cfg["bench"]
    sx, sy = float(bc["size_xy"][0]) / 2, float(bc["size_xy"][1]) / 2
    th, fz = float(bc["thickness"]), float(bc["floor_z"])
    spec.add_material(name="mat_frame", rgba=[0.90, 0.90, 0.88, 1.0], specular=0.3, shininess=0.4)
    spec.add_material(name="mat_shelf", rgba=[0.50, 0.50, 0.49, 1.0])
    spec.add_material(name="mat_wall", rgba=[0.62, 0.62, 0.60, 1.0])
    spec.add_material(name="mat_floor_f", rgba=[0.36, 0.36, 0.37, 1.0])
    vis = dict(contype=0, conaffinity=0, group=2)
    leg = 0.02
    for i, (cx, cy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        x, y = cx * (sx - 0.04), cy * (sy - 0.04)
        h = (-th - fz) / 2
        w.add_geom(name="bench_leg_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=[leg, leg, h], pos=[x, y, fz + h], material="mat_frame", **vis)
    for i, (x, y, hx, hy) in enumerate([(0, -(sy - 0.04), sx - 0.04, 0.015), (0, sy - 0.04, sx - 0.04, 0.015),
                                        (-(sx - 0.04), 0, 0.015, sy - 0.04), (sx - 0.04, 0, 0.015, sy - 0.04)]):
        w.add_geom(name="bench_apron_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.04], pos=[x, y, -th - 0.04], material="mat_frame", **vis)
    w.add_geom(name="bench_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx - 0.06, sy - 0.06, 0.01], pos=[0, 0, fz + 0.25], material="mat_shelf", **vis)
    for i, (x, y, hx, hy) in enumerate([(3.6, 0.0, 0.1, 4.0), (-3.2, 0.0, 0.1, 4.0), (0.2, 3.4, 4.0, 0.1), (0.2, -3.4, 4.0, 0.1)]):
        w.add_geom(name="backdrop_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 2.2], pos=[x, y, fz + 2.2], material="mat_wall", **vis)
    for g in spec.geoms:
        if g.name == "floor":
            g.material = "mat_floor_f"
    for name, (parent, T, ref) in board_poses(server).items():
        if parent == "world":
            body = spec.worldbody
        else:
            # the exact name over the copied spec's bodies (MjSpec.body() on a copied spec with an attached robot can
            # resolve a prefixed name to the wrong body: assembly_info/cell_render.py)
            body = [b for b in spec.bodies if b.name == parent][0]
        attach_ref(spec, body, ref, T)
    for t in spec.textures:                              # the boards' printed squares are sRGB images (rendering.md 3a)
        if (t.name or "").startswith(BOARD_GEOMS):
            t.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
    for l in list(spec.lights):                          # the episode's lights replace the cell's own
        spec.delete(l)
    return spec


def open_session(server):
    """Open the episode's render session (MuJoCo backend; sensor depth; one pass per frame)."""
    from render.backends import get_backend, require_capabilities
    from render.spec import BackgroundSpec, HeadlightSpec, LightSpec, MaterialOverride, SceneSpec
    be = get_backend("mujoco")
    require_capabilities(be, sessions=True, outputs=["rgb", "sensor_depth"])
    key = "hand-episode-%d-%d" % (id(server), int(server.ep_seed) & 0xFFFFFFFF)
    _SPECS.clear()
    _SPECS[key] = render_spec(server)
    L = server.episode["lights"]
    lc = server.episode["config"]["lights"]
    lights = []
    for k in ("key", "fill"):
        pos = np.asarray(lc[k]["pos"], float)
        d = np.asarray(lc[k]["target"], float) - pos
        d /= np.linalg.norm(d)
        lights.append(LightSpec(name="light_" + k, pos=pos.tolist(), dir=d.tolist(), diffuse=[L[k]] * 3, specular=[0.2 * L[k]] * 3,
                                castshadow=(k == "key"), cutoff_deg=70.0, exponent=2.0))
    ph = server.cfg["photo"]
    sc = SceneSpec(builder="arm.assembly_hand.render_twin:scene", builder_args={"key": key},
                   content_files=[os.path.join(server.kit_dir, "meshes")],
                   materials=[MaterialOverride(match="mat_bench", rgba=[0.74, 0.62, 0.46, 1.0], specular=0.2)],
                   lights=lights, replace_lights=True, headlight=HeadlightSpec(ambient=[L["dome"]] * 3, diffuse=[0.08] * 3, specular=[0.02] * 3),
                   background=BackgroundSpec(rgb1=[0.78, 0.78, 0.79], rgb2=[0.42, 0.42, 0.43]), shadows=True,
                   shadowsize=int(ph.get("shadowsize", 2048)), lod=str(ph.get("lod", "full")), near_m=0.01)
    return be.open(sc)


def triangle_count(spec):
    """the triangles of the twin's user meshes (for the frame-time readings)"""
    n = 0
    for m in spec.meshes:
        n += len(m.userface) // 3
    return n
