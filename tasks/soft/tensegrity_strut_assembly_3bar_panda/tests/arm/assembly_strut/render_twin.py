"""Render twin of the strut assembly cell for MuJoCo's classic renderer (the agent cameras' fallback renderer and the
base of the Filament scene, agent_render.py; written after the quadruped cell's render_twin.py).

The twin is the episode's physics MjSpec (same bodies and joint names: `session.sync_from(model, data)` copies the
state by joint name) plus visual-only additions: every module's render meshes (the authors' triangles from the hidden
kit's meshes/visual/, by material; ONE mesh asset per build, shared by every unit and strut of that build), the bench
frame, floor and backdrop walls, the three datum boards (procedural checkerboards of `render.scale_refs`: the bench
board on the world, the cradle board flat on the cradle plate's +x edge, the hand plate on the Panda hand) and the
episode's lights. Collision geoms (group 3) stay in the spec; the renderer draws groups 0-2. One render session per
episode (`open_session`).
"""

import json
import math
import os

import numpy as np

from ..assembly_info.cell_render import attach_ref, board_ref

__all__ = ["scene", "open_session", "render_spec", "board_poses", "add_visuals", "BOARD_GEOMS"]

_SPECS = {}
BOARD_GEOMS = ("bench_board", "cradle_board", "hand_plate")


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
    cb = FX.merged(fc)["board"]
    out["cradle_board"] = ("cradle", FX.board_pose_in_cradle(fc), board_ref("cradle_board", cb["cols"], cb["rows"], cb["square_mm"],
                                                                              cb["margin_mm"], cb["thickness_mm"]))
    h = HAND_PLATE
    T = _frame(h["normal_in_hand"], h["cols_along_in_hand"])
    T[:3, 3] = np.asarray(h["pos_in_hand_m"], float)
    J = np.asarray(server.episode["hand_plate_jitter"], float)
    out["hand_plate"] = ("panda/hand", T @ J, board_ref("hand_plate", h["cols"], h["rows"], h["square_mm"], h["margin_mm"], h["thickness_mm"]))
    return out


def _materials(spec, kit_info, units):
    """one material per colour key of kit_info.json (the photo material table) + the covers' colours"""
    cols = (kit_info or {}).get("colours") or {}
    made = {}
    have = set(mm.name for mm in spec.materials)
    for key, c in cols.items():
        if "kit_" + key in have:                       # a second call on the same spec (more bodies): reuse
            made[key] = "kit_" + key
            continue
        m = spec.add_material(name="kit_" + key, rgba=list(c.get("rgba", [0.5, 0.5, 0.5, 1.0])))
        m.specular = 0.3 if float(c.get("metallic", 0.0)) < 0.5 else 0.8
        m.shininess = float(max(0.05, 1.0 - float(c.get("roughness", 0.6))))
        made[key] = m.name
    return made


def piece_face_ranges(kit, build):
    """{material: [(piece, first face, face count)]}: the render meshes are the pieces' triangles concatenated in the
    order of kit.json's `pieces` list (as the kit is built); the counts from the faces of each piece's own file are not
    in the kit, so they come from `piece_triangles` ({piece: triangles}, e.g. render_omit.json's)"""
    out = {}
    for v in kit.modules[build]["visual"]:
        out[v["material"]] = list(v.get("pieces") or [])
    return out


def _omit_faces(F, pieces, tri, omit):
    """F without the faces of the pieces in `omit` (pieces in order, `tri` = {piece: triangle count})"""
    keep, k = [], 0
    for p in pieces:
        n = int(tri[p])
        if p not in omit:
            keep.append(F[k:k + n])
        k += n
    if k != len(F):
        raise ValueError("piece triangle counts %d != the mesh's %d faces" % (k, len(F)))
    return np.vstack(keep) if keep else F[:0]


def add_visuals(spec, server, omit=None, targets=None):
    """the authors' triangles on every module body of the twin (visual only, group 2, no mass); one mesh asset per
    (build, material); the end cap covers take their unit's colour. `omit` = {"pieces": {build: set of piece ids},
    "triangles": {build: {piece: triangle count}}}: those whole pieces are left out of the render meshes (the
    fully hidden blocks; no piece is changed); `targets` = [(body name, build, unit or None)] (default: the cell's)"""
    import mujoco
    kit = server.kit
    kinfo = server.kit_info or {}
    mats = _materials(spec, kinfo, server.cell.loose)
    cover = (kinfo.get("cover_colour_of_unit") or {})
    meshes = {mm.name: mm for mm in spec.meshes if (mm.name or "").startswith("vis_")}
    bodies = {b.name: b for b in spec.bodies}
    if targets is None:
        targets = [(c, "M0", None) for c in server.cell.carriers.values()] + [(u, kit.units[u]["build"], u) for u in server.cell.loose]
    for bname, build, unit in targets:
        body = bodies[bname]
        order = piece_face_ranges(kit, build)
        for mat, V, F in kit.visual_meshes(build):
            mn = "vis_%s_%s" % (build, mat)
            if omit and build in omit.get("pieces", {}):
                F = _omit_faces(F, order[mat], omit["triangles"][build], omit["pieces"][build])
                if len(F) == 0:
                    continue
            if mn not in meshes:
                ms = spec.add_mesh(name=mn)
                if len(V) < 4:
                    # MuJoCo needs >= 4 vertices per mesh: a material group of one triangle (M1a / M1b plastic_black)
                    # gets copies of its first vertex, unreferenced (no triangle is added or changed)
                    V = np.vstack([V] + [V[:1]] * (4 - len(V)))
                ms.uservert = (V * 1e-3).flatten().tolist()
                ms.userface = F.flatten().tolist()
                ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                meshes[mn] = ms
            key = cover.get(unit or bname, mat) if mat == "silicone" else mat
            kw = {"material": mats[key]} if key in mats else {"rgba": [0.6, 0.6, 0.6, 1.0]}
            body.add_geom(name="%s_vis_%s" % (bname, mat), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, contype=0, conaffinity=0, group=2,
                          mass=0.0, **kw)
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
    for l in list(spec.lights):                          # the episode's lights replace the cell's own
        spec.delete(l)
    return spec


def open_session(server):
    """Open the episode's render session (MuJoCo backend; sensor depth; one pass per frame)."""
    from render.backends import get_backend, require_capabilities
    from render.spec import BackgroundSpec, HeadlightSpec, LightSpec, MaterialOverride, SceneSpec
    be = get_backend("mujoco")
    require_capabilities(be, sessions=True, outputs=["rgb", "sensor_depth"])
    key = "strut-episode-%d-%d" % (id(server), int(server.ep_seed) & 0xFFFFFFFF)
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
    sc = SceneSpec(builder="arm.assembly_strut.render_twin:scene", builder_args={"key": key},
                   content_files=[os.path.join(server.kit_dir, "meshes")],
                   materials=[MaterialOverride(match="mat_table", rgba=[0.55, 0.55, 0.53, 1.0], specular=0.2)],
                   lights=lights, replace_lights=True, headlight=HeadlightSpec(ambient=[L["dome"]] * 3, diffuse=[0.08] * 3, specular=[0.02] * 3),
                   background=BackgroundSpec(rgb1=[0.78, 0.78, 0.79], rgb2=[0.42, 0.42, 0.43]), shadows=True,
                   shadowsize=int(ph.get("shadowsize", 2048)), lod=str(ph.get("lod", "full")), near_m=0.01)
    return be.open(sc)


def triangle_count(spec):
    """the triangles of the twin's visual meshes (for the frame-time readings)"""
    n = 0
    for m in spec.meshes:
        f = m.userface
        n += len(f) // 3
    return n
