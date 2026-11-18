"""Render twin of the quadruped assembly cell for MuJoCo's classic renderer (the agent cameras' fallback renderer and the
base of the Filament scene, agent_render.py; a port of assembly_info/cell_render.py).

The twin is the episode's physics MjSpec (same bodies and joint names: `session.sync_from(model, data)` copies the
state by joint name) plus visual-only additions: the robot table's frame under the 2.0 x 1.4 m top, the floor and
backdrop walls, the three datum boards (procedural checkerboards of `render.scale_refs`: the bench board on the world,
the stand board on the stand rail's face towards the robot arm, the tool plate on the camera housing's outer face) and
the episode's lights. One render session per episode (`open_session`).
"""

import os

import numpy as np

from ..assembly_info.cell_render import attach_ref, board_ref

__all__ = ["scene", "open_session", "render_spec", "board_poses", "BOARD_GEOMS"]

_SPECS = {}
BOARD_GEOMS = ("bench_board", "stand_board", "tool_plate")


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


def _body_T(m, d, name):
    b = m.body(name).id
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def board_poses(server):
    """{name: (parent body, T_in_parent, ref)} of the three boards for this episode"""
    import math
    from .server import BENCH_BOARD, STAND_BOARD, TOOL_PLATE
    m, d = server.model, server.data
    cfg = server.cell.cfg
    out = {}
    b = BENCH_BOARD
    T = _frame([0, 0, 1], [math.cos(math.radians(b["yaw_deg"])), math.sin(math.radians(b["yaw_deg"])), 0])
    T[:3, 3] = b["pos"]
    out["bench_board"] = ("world", T, board_ref("bench_board", b["cols"], b["rows"], b["square_mm"]))
    # the stand board: on the rail's face towards the arm (the stand frame's +y at yaw 90 deg = the cell's -x)
    s = STAND_BOARD
    st = cfg["stand"]
    zb = -float(st["body_z"])
    T = _frame([0, 1, 0], [1, 0, 0])
    T[:3, 3] = [0.0, float(st["post"]) / 2 + 0.0005, zb + float(st["rail_z"])]
    out["stand_board"] = ("stand", T, board_ref("stand_board", s["cols"], s["rows"], s["square_mm"], s["margin_mm"], s["thickness_mm"]))
    # the tool plate: on the camera housing's outer face (the tool's +x side), columns along the tool axis
    t = TOOL_PLATE
    rb = server.cell.robot
    hg = m.geom(rb["prefix"] + "wrist_cam_housing").id
    mount = m.body(int(m.geom_bodyid[hg])).name
    Tm = _body_T(m, d, mount)
    site = m.site(rb["tcp_site"]).id
    R_tcp = d.site_xmat[site].reshape(3, 3)
    n_m = Tm[:3, :3].T @ R_tcp[:, 0]                       # the tool's +x in the mount body's frame
    z_m = Tm[:3, :3].T @ R_tcp[:, 2]                       # the tool axis
    gp = np.asarray(m.geom_pos[hg], float)
    gq = np.asarray(m.geom_quat[hg], float)
    Rg = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(Rg, gq)
    Rg = Rg.reshape(3, 3)
    half = np.asarray(m.geom_size[hg][:3], float)
    k = int(np.argmax(np.abs(Rg.T @ n_m)))
    sign = float(np.sign((Rg.T @ n_m)[k]))
    centre = gp + Rg[:, k] * sign * half[k]
    T = _frame(n_m, z_m)
    T[:3, 3] = centre + n_m * 0.0005
    J = np.asarray(server.episode["hand_plate_jitter"], float)
    out["tool_plate"] = (mount, T @ J, board_ref("tool_plate", t["cols"], t["rows"], t["square_mm"], t["margin_mm"], t["thickness_mm"]))
    return out


def render_spec(server):
    """The episode's render MjSpec (a copy of the physics spec + visual additions)."""
    import mujoco
    spec = server.cell.spec.copy()
    w = spec.worldbody
    bc = server.cell.cfg["bench"]
    cx, cy = (float(v) for v in bc["centre_xy"])
    sx, sy = float(bc["size_xy"][0]) / 2, float(bc["size_xy"][1]) / 2
    th, fz = float(bc["thickness"]), float(bc["floor_z"])
    spec.add_material(name="mat_frame", rgba=[0.22, 0.23, 0.25, 1.0], specular=0.3, shininess=0.4)
    spec.add_material(name="mat_shelf", rgba=[0.30, 0.31, 0.32, 1.0])
    spec.add_material(name="mat_wall", rgba=[0.62, 0.62, 0.60, 1.0])
    spec.add_material(name="mat_floor_f", rgba=[0.36, 0.36, 0.37, 1.0])
    vis = dict(contype=0, conaffinity=0, group=2)
    leg = 0.03
    k = 0
    for ix in (-1, 0, 1):
        for iy in (-1, 1):
            x, y = cx + ix * (sx - 0.06), cy + iy * (sy - 0.06)
            h = (-th - fz) / 2
            w.add_geom(name="bench_leg_%d" % k, type=mujoco.mjtGeom.mjGEOM_BOX, size=[leg, leg, h], pos=[x, y, fz + h], material="mat_frame", **vis)
            k += 1
    for i, (x, y, hx, hy) in enumerate([(cx, cy - (sy - 0.06), sx - 0.06, 0.03), (cx, cy + sy - 0.06, sx - 0.06, 0.03),
                                        (cx - (sx - 0.06), cy, 0.03, sy - 0.06), (cx + sx - 0.06, cy, 0.03, sy - 0.06)]):
        w.add_geom(name="bench_rail_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.03], pos=[x, y, -th - 0.07], material="mat_frame", **vis)
    w.add_geom(name="bench_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx - 0.09, sy - 0.09, 0.006], pos=[cx, cy, fz + 0.2], material="mat_shelf", **vis)
    for i, (x, y, hx, hy) in enumerate([(cx + 3.4, cy, 0.1, 4.0), (cx - 3.4, cy, 0.1, 4.0), (cx, cy + 3.4, 4.0, 0.1), (cx, cy - 3.4, 4.0, 0.1)]):
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
    sc = SceneSpec(builder="arm.assembly_quad.render_twin:scene", builder_args={"key": key},
                   content_files=[os.path.join(server.kit_dir, "meshes")],
                   materials=[MaterialOverride(match="mat_bench", rgba=[0.62, 0.52, 0.38, 1.0], specular=0.2)],
                   lights=lights, replace_lights=True, headlight=HeadlightSpec(ambient=[L["dome"]] * 3, diffuse=[0.08] * 3, specular=[0.02] * 3),
                   background=BackgroundSpec(rgb1=[0.78, 0.78, 0.79], rgb2=[0.42, 0.42, 0.43]), shadows=True,
                   shadowsize=int(ph.get("shadowsize", 2048)), lod=str(ph.get("lod", "full")), near_m=0.01)
    return be.open(sc)
