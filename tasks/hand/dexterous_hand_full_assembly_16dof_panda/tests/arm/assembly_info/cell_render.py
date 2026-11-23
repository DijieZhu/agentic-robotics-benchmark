"""Render twin of the information-tier assembly cell (assembly.md 11.2; PUBLIC CODE).

The live cameras of `cell_info.InfoCellServer` are rendered through the renderer-agnostic layer `shared/render/`
(T3): one `RenderBackend.open(scene)` session per episode on a render twin of the physics cell. The twin is the episode's
physics MjSpec (same bodies and joint names, so `session.sync_from(model, data)` copies the state by joint name) plus
visual-only additions: the workbench frame, the factory floor and backdrop walls, the three datum boards (physical bodies
with procedural textures from `render.scale_refs`, so any backend renders them: the bench board on the world, the fixture
board on the fixture, the hand plate on the Panda hand) and the episode's lights. The builder hands the render layer a
copy of the MjSpec registered for this process (`scene(key=...)`); its request digest covers the builder source and the
kit's mesh folder.
"""

import math
import os

import numpy as np

__all__ = ["scene", "open_session", "render_spec", "board_ref", "attach_ref", "BOARD_GEOMS"]

_SPECS = {}
BOARD_GEOMS = ("bench_board", "fixture_board", "hand_plate")


def scene(key):
    """SceneSpec builder: a fresh copy of the registered render MjSpec."""
    return _SPECS[key].copy()


def board_ref(name, cols, rows, square_mm, margin_mm=None, thickness_mm=3.0):
    from render import scale_refs as SR
    params = {"cols": int(cols), "rows": int(rows), "square_mm": float(square_mm), "thickness_mm": float(thickness_mm)}
    if margin_mm is not None:
        params["margin_mm"] = float(margin_mm)
    return SR.checkerboard(name, pos=(0.0, 0.0, 0.0), **params)


def attach_ref(spec, parent, ref, T_in_parent):
    """Attach a render-layer scale reference (its `parts()`: boxes / textured meshes) to `parent` (an MjsBody) at
    T_in_parent (4x4, the ref frame = the centre of the face it rests on, +z out of it). Visual only (group 2, no contacts,
    no mass). Returns the geom names."""
    import mujoco
    from render import scale_refs as SR
    from ..assembly_six.layout import quat_from_rotmat
    pre = ref.name + "__"
    child = parent.add_body(name=ref.name, pos=list(np.asarray(T_in_parent, float)[:3, 3]), quat=quat_from_rotmat(np.asarray(T_in_parent, float)[:3, :3]))
    names = []
    for part in SR.parts(ref):
        mat = spec.add_material(name=pre + part.name, rgba=list(part.rgba), specular=part.specular, shininess=part.shininess,
                                reflectance=part.reflectance)
        if part.texture is not None:
            h, w = part.texture.shape[:2]
            tex = spec.add_texture(name=pre + part.name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=3)
            tex.data = np.ascontiguousarray(part.texture, np.uint8).tobytes()
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        kw = dict(name=pre + part.name, material=pre + part.name, contype=0, conaffinity=0, group=2, pos=list(part.pos), mass=0.0)
        if part.kind == "mesh":
            m = part.mesh
            mesh = spec.add_mesh(name=pre + part.name + "_mesh", uservert=m.vertices.reshape(-1).tolist(), userface=m.faces.reshape(-1).tolist())
            if m.uv is not None:
                mesh.usertexcoord = m.uv.reshape(-1).tolist()
                mesh.userfacetexcoord = m.faces.reshape(-1).tolist()
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            child.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, **kw)
        elif part.kind == "box":
            child.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in part.size], **kw)
        else:
            child.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[part.size[0], part.size[1] / 2, 0], **kw)
        names.append(pre + part.name)
    return names


def _frame(z, x):
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - z * float(np.dot(x, z))
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    return T


def board_poses(server):
    """World / parent poses of the three boards for this episode: {name: (parent body, T_in_parent, ref)}."""
    from .cell_info import BENCH_BOARD, FIXTURE_BOARD, HAND_PLATE
    out = {}
    b = BENCH_BOARD
    T = np.eye(4)
    T[:3, :3] = _frame([0, 0, 1], [math.cos(math.radians(b["yaw_deg"])), math.sin(math.radians(b["yaw_deg"])), 0])[:3, :3]
    T[:3, 3] = b["pos"]
    out["bench_board"] = ("world", T, board_ref("bench_board", b["cols"], b["rows"], b["square_mm"]))
    f = FIXTURE_BOARD
    ref = board_ref("fixture_board", f["cols"], f["rows"], f["square_mm"], f["margin_mm"], f["thickness_mm"])
    by = (f["rows"] * f["square_mm"] + 2 * f["margin_mm"]) * 1e-3
    half = float(server.meta["config"]["fixture"]["size_m"]) / 2
    T = _frame([1, 0, 0], [0, 1, 0])
    T[:3, 3] = [half + 0.0005, 0.0, by / 2]
    out["fixture_board"] = ("fixture", T, ref)
    h = HAND_PLATE
    ref = board_ref("hand_plate", h["cols"], h["rows"], h["square_mm"], h["margin_mm"], h["thickness_mm"])
    T = _frame(h["normal_in_hand"], h["cols_along_in_hand"])
    T[:3, 3] = np.asarray(h["pos_in_hand_m"], float) - np.asarray(h["normal_in_hand"], float) * 0.0   # base face on the housing
    J = np.asarray(server.episode["hand_plate_jitter"], float)
    out["hand_plate"] = ("panda/hand", T @ J, ref)
    return out


def render_spec(server):
    """The episode's render MjSpec (a copy of the physics spec + visual additions)."""
    import mujoco
    spec = server.build.spec.copy()
    w = spec.worldbody
    tb = server.cfg["cell"].get("table", {"size_xy": [1.6, 1.0], "thickness": 0.03, "floor_z": -0.994})
    sx, sy = tb["size_xy"][0] / 2, tb["size_xy"][1] / 2
    th, fz = float(tb["thickness"]), float(tb["floor_z"])
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
            # the body by an exact name match over the (copied) spec's bodies: MjSpec.body() on a copied spec with an
            # attached (prefixed) robot resolved "panda/hand" to its parent link (the plate then hung 0.107 m behind the hand)
            body = [b for b in spec.bodies if b.name == parent][0]
        attach_ref(spec, body, ref, T)
    return spec


def open_session(server):
    """Open the episode's render session (MuJoCo backend; RA1 sensor depth, RA2 one pass, RA4 sessions)."""
    from render.backends import get_backend, require_capabilities
    from render.spec import BackgroundSpec, HeadlightSpec, LightSpec, MaterialOverride, SceneSpec
    be = get_backend("mujoco")
    require_capabilities(be, sessions=True, outputs=["rgb", "sensor_depth"])
    key = "episode-%d-%d" % (id(server), int(server.ep_seed) & 0xFFFFFFFF)
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
    sc = SceneSpec(builder="arm.assembly_info.cell_render:scene", builder_args={"key": key},
                   content_files=[os.path.join(server.a2_dir, "meshes", "visual")],
                   materials=[MaterialOverride(match="mat_table", rgba=[0.55, 0.55, 0.53, 1.0], specular=0.2)],
                   lights=lights, replace_lights=True, headlight=HeadlightSpec(ambient=[L["dome"]] * 3, diffuse=[0.08] * 3, specular=[0.02] * 3),
                   background=BackgroundSpec(rgb1=[0.78, 0.78, 0.79], rgb2=[0.42, 0.42, 0.43]), shadows=True,
                   shadowsize=int(ph.get("shadowsize", 2048)), lod=str(ph.get("lod", "full")), near_m=0.01)
    return be.open(sc)
