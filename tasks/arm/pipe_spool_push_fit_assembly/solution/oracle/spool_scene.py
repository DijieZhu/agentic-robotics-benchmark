"""Pipe spool push-fit assembly: what the cameras draw (plain data: meshes, materials, frames; no renderer here).

One scene, two renderers: these are the demonstration pictures' own meshes. Every part of the simulated model is drawn
at the model's own geometry, read from the episode's model and spool_sim's numbers: the pipes (copper tube at the
model's radius and the episode's lengths, pipe A's band), the elbows inside their collision shapes (the unit's bore
clearance; nothing shows the unit's stop depth), both Pandas' visual meshes with the gripper's soft pads. Added to
them (spool_parts.py): the bench as a worktop on a steel frame in the table block's place, an adapter plate under
each robot, the camera rig (the model has contact boxes of the same size: spool_parts.contact_shapes), the stock on
the bench's lower shelf. The hall round the bench is the render bundle's.

`parts(unit, episode, model)`: [(body name or "world", part name, mesh (V, N, UV, F) in that body's frame, look,
options)], in a fixed order. `items(unit, episode, model)`: the same merged per frame and material for the
renderer, each a dict: frame ("world" | ("body", id)), look, mesh (float32 V, N, UV already divided by the look's
tile, uint32 F), mat (rgb sRGB 0..1, roughness, metallic, emissive), tex (None | ("static", name)).
"""
import math
import os

import mujoco
import numpy as np

import spool_geom as G
import spool_looks as LK
import spool_parts as P

RENDER_DIR = os.environ.get("SPOOL_RENDER_DIR", "/opt/task-assets/payload/render")
PROUD = 0.0004                    # m: a robot's dark parts stand this far off the shell they lie on


PALETTE = ("panda_white", "panda_black", "panda_grey", "panda_blue", "panda_led_green", "panda_led_red")


def palette_image():
    """the Pandas' palette map: one 4 x 4 block of each look's colour in a row (sRGB, uint8)"""
    img = np.zeros((4, 4 * len(PALETTE), 3), np.uint8)
    for k, look in enumerate(PALETTE):
        img[:, 4 * k:4 * k + 4] = np.round(255.0 * _srgb(LK.LOOKS[look]["base"])).astype(np.uint8)
    return img


def _srgb(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def texture_name(look):
    """the render bundle's name of a look's colour map (the map tinted to the look's mean), or None"""
    L = LK.LOOKS.get(look) or LK.LOOKS["default"]
    return ("%s__%s" % (L["col"][0], look)) if "col" in L else None


def material(look):
    """A look as the rasteriser's terms: (rgb sRGB, roughness, metallic, emissive, static texture name or None,
    metres per uv repeat or None for a map with its own uv)."""
    L = LK.LOOKS.get(look) or LK.LOOKS["default"]
    rough = L.get("rough")
    if rough is None:
        rough = (L.get("rgh") or (None, 0.5, 1.0))[1]
        rough = 0.5 if rough is None else rough
    tex = texture_name(look)
    rgb = np.ones(3) if tex else _srgb(L.get("base", [0.5, 0.5, 0.5]))
    tile = None if L.get("uv") == "own" else float(L.get("tile", 0.1))
    return (rgb, float(rough), float(L.get("metal", 0.0)), 0.0, tex, tile)


def quat_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def model_soup(m, g):
    """a mesh geom as a triangle soup with its normals, in its body's frame"""
    mid = int(m.geom_dataid[g])
    va = int(m.mesh_vertadr[mid])
    fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    Vm = m.mesh_vert[va:va + int(m.mesh_vertnum[mid])].astype(float)
    Fm = m.mesh_face[fa:fa + fn].astype(int)
    na = int(m.mesh_normaladr[mid])
    Nm = m.mesh_normal[na:na + int(m.mesh_normalnum[mid])].astype(float)
    FN = m.mesh_facenormal[fa:fa + fn].astype(int)
    R = quat_mat(m.geom_quat[g])
    return Vm[Fm.reshape(-1)] @ R.T + m.geom_pos[g], Nm[FN.reshape(-1)] @ R.T


def model_mesh(m, g):
    V, N = model_soup(m, g)
    return G.uv_box((V, N, np.zeros((len(V), 2)), np.arange(len(V)).reshape(-1, 3)))


def lift(mesh, d):
    """the mesh moved out by d along its surface: every vertex along the mean normal of all the vertices at its place
    (the faces carry their own vertices; moved along their own normals they would part at the edges)"""
    V, N, UV, F = mesh
    _u, inv = np.unique(np.round(V / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    S_ = np.zeros((int(inv.max()) + 1, 3))
    np.add.at(S_, inv, N)
    S_ /= np.linalg.norm(S_, axis=1, keepdims=True) + 1e-12
    return V + d * S_[inv], N, UV, F


def panda_look(m, g):
    rgba = m.mat_rgba[int(m.geom_matid[g])] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    if min(rgba[:3]) > 0.85:
        return "panda_white"
    if max(rgba[:3]) < 0.15:
        return "panda_black"
    if max(rgba[:3]) < 0.35 and max(rgba[:3]) - min(rgba[:3]) < 0.05:
        return "panda_grey"
    if rgba[0] > 0.9 and rgba[1] < 0.1:
        return "panda_led_red"
    if rgba[1] > 0.9 and rgba[0] < 0.1:
        return "panda_led_green"
    if rgba[2] > 0.7 and rgba[0] < 0.1:
        return "panda_blue"
    return "panda_white"


def camera_mesh():
    """the fixed cameras' body (the render bundle's props.npz: a machine-vision camera body, metres)"""
    z = np.load(os.path.join(RENDER_DIR, "props.npz"))
    return z["cam_V"].astype(float), z["cam_N"].astype(float), z["cam_UV"].astype(float), z["cam_F"].astype(np.int64)


def parts(unit, episode, m):
    import spool_sim as S
    out = []

    def put(body, name, mesh, look, opt=None):
        out.append((body, name, mesh, look, opt or {}))

    for name, mesh, look, opt in P.bench(S.TABLE_Z, S.TABLE_EDGE_X, S.TABLE_HALF_Y, [tuple(v) for v in S.BASES.values()]):
        put("world", name, mesh, look, opt)
    for name, mesh, look, opt in P.rig(S.CAMERAS, S.TABLE_Z, camera_mesh()):
        put("world", name, mesh, look, opt)
    # the Pandas: the model's visual meshes (group 1), the fingers' metal with soft pads in their tip blocks' place
    panda_dark, panda_white = [], []
    for g in range(m.ngeom):
        body = m.body(int(m.geom_bodyid[g])).name
        if not body.startswith(("l_", "r_")) or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH) or int(m.geom_group[g]) != 1:
            continue
        if body.endswith(("_g_leftfinger", "_g_rightfinger")):
            sg = 1.0 if body.endswith("leftfinger") else -1.0
            V, N = model_soup(m, g)
            put(body, "finger_metal", P.finger_metal(V, N, sg), "finger")
            put(body, "finger_pad", P.finger_pad(sg), "finger_pad")
        else:
            # A link's shells, bands and caps are ONE mesh with one material (look "panda": their colours from a small
            # palette map, one texel per look): where two of them lie in one surface the mesh's own triangle order
            # decides the pixel, the same in every process. The dark parts come first and stand PROUD of the shell.
            look = panda_look(m, g)
            V, N, _uv, F = model_mesh(m, g)
            if look != "panda_white":
                V, N, _uv, F = lift((V, N, _uv, F), PROUD)
            uv = np.tile([(PALETTE.index(look) + 0.5) / len(PALETTE), 0.5], (len(V), 1))
            (panda_dark if look != "panda_white" else panda_white).append((body, "panda_%d" % g, (V, N, uv, F)))
    for body, name, mesh in panda_dark + panda_white:
        put(body, name, mesh, "panda", {"uv": "own"})
    for s in ("l", "r"):
        cid = m.camera(s + "_eye_in_hand").id
        Rc = quat_mat(m.cam_quat[cid])
        Rb = np.stack([-Rc[:, 2], Rc[:, 0], Rc[:, 1]], 1)
        for name, mesh, look, opt in P.wrist_camera(Rb, m.cam_pos[cid].astype(float)):
            put(m.body(int(m.cam_bodyid[cid])).name, name, mesh, look, opt)
    for which, key in (("A", "len_a"), ("B", "len_b")):
        L = float(episode[key])
        band = episode["band"] * (L / 2 - S.BAND_FROM_END) if which == "A" else None
        for name, mesh, look, opt in P.pipe(L, S.R_PIPE, S.CHAMFER, band, S.BAND_HALF):
            put(which, name, mesh, look, opt)
    el = P.elbow(S.R_PIPE, float(unit["clear"]), S.SOCK_L, S.BORE_D, S.R_OUT, S.CORNER_HALF, S.OR_DEPTH)
    for e in (0, 1):
        for name, mesh, look, opt in el:
            put("E%d" % e, name, mesh, look, opt)
    for name, mesh, look, opt in P.stock(S.TABLE_Z, S.R_PIPE, P.elbow(S.R_PIPE, 0.0003, S.SOCK_L, S.BORE_D, S.R_OUT, S.CORNER_HALF, S.OR_DEPTH), S.CORNER_HALF[2]):
        put("world", name, mesh, look, opt)
    return out


class Collector(object):
    """Parts merged per frame and material, in the order they first appear."""

    def __init__(self):
        self.groups = {}

    def add(self, frame, look, mesh, opt=None):
        opt = opt or {}
        V, N, UV, F = [np.asarray(x) for x in mesh]
        if len(F) == 0:
            return
        rgb, rough, metal, emis, stex, tile = material(look)
        V, N, UV = V.astype(np.float64), N.astype(np.float64), UV.astype(np.float64)
        tex = ("static", stex) if stex is not None else None
        if look == "panda":
            tex = ("data", "panda_palette")
        if not (opt.get("uv") == "own" or tile is None):
            UV = UV / tile
        F = F.reshape(-1, 3).astype(np.int64)
        g = self.groups.setdefault((frame, look), {"frame": frame, "look": look, "tex": tex, "mat": (rgb, rough, metal, emis),
                                                   "V": [], "N": [], "U": [], "F": [], "nv": 0})
        g["V"].append(V)
        g["N"].append(N)
        g["U"].append(UV)
        g["F"].append(F + g["nv"])
        g["nv"] += len(V)

    def items(self):
        out = []
        for g in self.groups.values():
            N = np.concatenate(g["N"])
            N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
            out.append({"frame": g["frame"], "look": g["look"], "tex": g["tex"], "mat": g["mat"],
                        "tex_img": palette_image() if g["tex"] is not None and g["tex"][0] == "data" else None,
                        "mesh": (np.ascontiguousarray(np.concatenate(g["V"]), np.float32), np.ascontiguousarray(N, np.float32),
                                 np.ascontiguousarray(np.concatenate(g["U"]), np.float32),
                                 np.ascontiguousarray(np.concatenate(g["F"]).reshape(-1), np.uint32))})
        return out


def items(unit, episode, m):
    col = Collector()
    for body, _name, mesh, look, opt in parts(unit, episode, m):
        col.add("world" if body == "world" else ("body", int(m.body(body).id)), look, mesh, opt)
    return col.items()
