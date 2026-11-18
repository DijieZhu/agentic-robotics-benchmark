"""Shared helpers: poses and quaternions, MjSpec material / texture / mesh helpers, the TargetNames record.

Frame convention of every target ("face frame"): local +x points out of the readable face towards the viewer,
local +y is the viewer's right, local +z is up. The pose given to a target's `add_to_spec` places this frame; its
origin is the centre of the readable face (dial centre, valve stem axis on the pipe, level gauge window centre,
lamp row centre, tag plate centre).

Texture convention (measured on MuJoCo 3.14): a 2D texture on a box shows upright and unmirrored on the box's +x face
seen from +x (image left at -y, image top at +z); mesh texture coordinates have v = 0 at the image's top row.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = ["Pose", "TargetNames", "quat_from_mat", "mat_from_quat", "quat_mul", "quat_axis_angle", "rot_x", "rot_y",
           "rot_z", "look_at_R", "save_png", "texture_material", "color_material", "add_disc_mesh",
           "add_annulus_mesh", "add_torus_mesh", "font", "draw_centered_text", "fit_font", "QUAT_Z_TO_X",
           "QUAT_Z_TO_Y", "visual_only", "geom_ids", "body_subtree_ids", "set_offscreen", "opt_names", "hinge_range"]


# ----------------------------------------------------------------------------------------------- rotations
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def quat_from_mat(R) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a 3x3 rotation matrix."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def mat_from_quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


QUAT_Z_TO_X = quat_axis_angle([0, 1, 0], math.pi / 2)    # turns a geom's local z axis (cylinder axis) onto +x
QUAT_Z_TO_Y = quat_axis_angle([1, 0, 0], -math.pi / 2)   # turns local z onto +y


def look_at_R(forward, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose columns are (forward, left, up'): a body frame with +x along `forward` and +z as close to `up`
    as possible (the robot convention x forward, y left, z up)."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, float)
    left = np.cross(u, f)
    if np.linalg.norm(left) < 1e-9:
        left = np.cross([1.0, 0.0, 0.0] if abs(f[0]) < 0.9 else [0.0, 1.0, 0.0], f)
    left /= np.linalg.norm(left)
    up2 = np.cross(f, left)
    return np.stack([f, left, up2], 1)


@dataclass
class Pose:
    """Position (m) and orientation (unit quaternion w, x, y, z) of a frame relative to its parent."""
    pos: Sequence[float] = (0.0, 0.0, 0.0)
    quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def facing(pos, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> "Pose":
        """A face frame at `pos` whose normal (+x) points along azimuth `yaw_deg` (counter-clockwise from world +x,
        about +z) tilted up by `pitch_deg`; `roll_deg` turns the face about its own normal (counter-clockwise as the
        viewer sees it)."""
        R = rot_z(math.radians(yaw_deg)) @ rot_y(-math.radians(pitch_deg)) @ rot_x(math.radians(roll_deg))
        return Pose(tuple(float(v) for v in pos), tuple(quat_from_mat(R)))

    def R(self) -> np.ndarray:
        return mat_from_quat(self.quat)

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R()
        T[:3, 3] = self.pos
        return T

    def compose(self, other: "Pose") -> "Pose":
        """self * other (other expressed in self's frame)."""
        p = np.asarray(self.pos, float) + self.R() @ np.asarray(other.pos, float)
        return Pose(tuple(p), tuple(quat_mul(self.quat, other.quat)))

    def apply(self, p_local) -> np.ndarray:
        return np.asarray(self.pos, float) + self.R() @ np.asarray(p_local, float)


@dataclass
class TargetNames:
    """What a target's add_to_spec created (all names are prefixed '<name>/').

    face_geoms: the geoms that carry the reading (dial, lever, indicator, glass, lenses, marker plate); a grader can
    count their pixels in a segmentation image to decide whether the target was in view.
    state: the bookkeeping set_state needs (joint names, geom names, stroke, ...)."""
    name: str
    kind: str
    body: str
    geoms: list = field(default_factory=list)
    face_geoms: list = field(default_factory=list)
    bodies: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    textures: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    tag: "TargetNames | None" = None

    def all_geoms(self) -> list:
        return list(self.geoms) + (self.tag.all_geoms() if self.tag is not None else [])


def opt_names(model, objtype, names: Iterable[str]) -> np.ndarray:
    import mujoco
    return np.array([mujoco.mj_name2id(model, objtype, n) for n in names], np.int32)


def geom_ids(model, names: Iterable[str]) -> np.ndarray:
    import mujoco
    ids = opt_names(model, mujoco.mjtObj.mjOBJ_GEOM, names)
    if (ids < 0).any():
        bad = [n for n, i in zip(names, ids) if i < 0]
        raise KeyError("unknown geoms: %s" % bad[:5])
    return ids


def body_subtree_ids(model, root_body_id: int) -> np.ndarray:
    """Ids of a body and all its descendants."""
    ids = [int(root_body_id)]
    par = np.asarray(model.body_parentid)
    changed = True
    s = set(ids)
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in s and int(par[b]) in s and b != 0:
                s.add(b)
                changed = True
    return np.array(sorted(s), np.int32)


def set_offscreen(spec, width: int = 1280, height: int = 960, znear: float = 0.01, zfar: float = 60.0,
                  extent: float | None = None, headlight: float = 0.0, shadowsize: int = 2048):
    """Offscreen buffer size, clip planes (absolute metres; MuJoCo stores them relative to the model extent) and the
    headlight (off by default: lights come from the scene and the robot's lamp)."""
    ext = float(extent) if extent is not None else 10.0
    spec.stat.extent = ext
    spec.visual.global_.offwidth = int(width)
    spec.visual.global_.offheight = int(height)
    spec.visual.map.znear = znear / ext
    spec.visual.map.zfar = zfar / ext
    spec.visual.headlight.ambient = [headlight * 0.3] * 3
    spec.visual.headlight.diffuse = [headlight] * 3
    spec.visual.headlight.specular = [0.0] * 3
    spec.visual.quality.shadowsize = int(shadowsize)


def hinge_range(spec, lo_deg: float, hi_deg: float) -> list:
    """A hinge joint range in the unit the spec's compiler expects. MjSpec, like MJCF, defaults to
    `compiler.degree = True`, so a range written in radians would be read as degrees (a +-140 deg needle stop became
    +-2.4 deg and the needle sprang back to mid-scale as soon as the world stepped)."""
    if bool(getattr(spec.compiler, "degree", True)):
        return [float(lo_deg), float(hi_deg)]
    return [math.radians(lo_deg), math.radians(hi_deg)]


def visual_only(kw=None, group=1) -> dict:
    """Keyword arguments for a geom that is drawn and seen by rays but takes no part in contacts."""
    d = dict(kw or {})
    d.update(contype=0, conaffinity=0, group=group)
    return d


# ----------------------------------------------------------------------------------------------- textures / materials
def save_png(img: np.ndarray, assets_dir: str, prefix: str) -> str:
    """Write an RGB(A) uint8 image as PNG under assets_dir with a content-hashed file name (identical images share a
    file); returns the absolute path."""
    from PIL import Image
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h = hashlib.sha1(img.tobytes() + str(img.shape).encode()).hexdigest()[:12]
    os.makedirs(assets_dir, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix)
    path = os.path.abspath(os.path.join(assets_dir, "%s_%s.png" % (safe, h)))
    if not os.path.exists(path):
        tmp = path + ".tmp%d" % os.getpid()
        Image.fromarray(img).save(tmp, format="PNG", optimize=False)
        os.replace(tmp, path)
    return path


def _has(spec, kind: str, name: str) -> bool:
    try:
        return getattr(spec, kind)(name) is not None
    except Exception:  # noqa: BLE001
        return False


def texture_material(spec, name: str, png_path: str, specular: float = 0.15, shininess: float = 0.2,
                     rgba=(1, 1, 1, 1), emission: float = 0.0) -> str:
    """A 2D texture from a PNG file plus a material that shows it (reflectance 0); returns the material name.
    Reuses the material if it already exists."""
    import mujoco
    if _has(spec, "material", name):
        return name
    tname = name + "/tex"
    if not _has(spec, "texture", tname):
        spec.add_texture(name=tname, type=mujoco.mjtTexture.mjTEXTURE_2D, file=png_path)
    m = spec.add_material(name=name, rgba=list(rgba), specular=specular, shininess=shininess, reflectance=0.0,
                          emission=emission)
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tname
    return name


def color_material(spec, name: str, rgba, specular: float = 0.3, shininess: float = 0.3, emission: float = 0.0) -> str:
    if _has(spec, "material", name):
        return name
    rgba = list(rgba) + [1.0] * (4 - len(rgba))
    spec.add_material(name=name, rgba=rgba, specular=specular, shininess=shininess, reflectance=0.0,
                      emission=emission)
    return name


# ----------------------------------------------------------------------------------------------- meshes
def _add_mesh(spec, name, verts, faces, texcoord=None):
    m = spec.add_mesh(name=name)
    m.uservert = np.asarray(verts, float).ravel().tolist()
    m.userface = np.asarray(faces, np.int32).ravel().tolist()
    if texcoord is not None:
        m.usertexcoord = np.asarray(texcoord, float).ravel().tolist()
    return name


def add_disc_mesh(spec, name: str, radius: float, thickness: float, n: int = 72) -> str:
    """A closed thin disc (prism) facing +x, front face at x = 0, back at x = -thickness; the front face carries
    texture coordinates mapping the image's inscribed circle (image right = +y, image top = +z)."""
    if _has(spec, "mesh", name):
        return name
    verts, tc, faces = [[0.0, 0.0, 0.0]], [[0.5, 0.5]], []
    for k in range(n):
        a = 2 * math.pi * k / n
        verts.append([0.0, radius * math.cos(a), radius * math.sin(a)])
        tc.append([0.5 + 0.5 * math.cos(a), 0.5 - 0.5 * math.sin(a)])
    for k in range(n):
        faces.append([0, 1 + k, 1 + (k + 1) % n])
    b0 = len(verts)
    verts.append([-thickness, 0.0, 0.0])
    tc.append([0.5, 0.5])
    for k in range(n):
        a = 2 * math.pi * k / n
        verts.append([-thickness, radius * math.cos(a), radius * math.sin(a)])
        tc.append([0.5 + 0.5 * math.cos(a), 0.5 - 0.5 * math.sin(a)])
    for k in range(n):
        faces.append([b0, b0 + 1 + (k + 1) % n, b0 + 1 + k])
    for k in range(n):
        a0, a1 = 1 + k, 1 + (k + 1) % n
        c0, c1 = b0 + 1 + k, b0 + 1 + (k + 1) % n
        faces.append([a0, c0, c1])
        faces.append([a0, c1, a1])
    return _add_mesh(spec, name, verts, faces, tc)


def add_annulus_mesh(spec, name: str, r_in: float, r_out: float, x_front: float, x_back: float, n: int = 72) -> str:
    """A closed ring (washer) about the x axis between x_back and x_front: a gauge bezel, a lamp bezel."""
    if _has(spec, "mesh", name):
        return name
    verts, faces = [], []
    for xx in (x_front, x_back):
        for r in (r_in, r_out):
            for k in range(n):
                a = 2 * math.pi * k / n
                verts.append([xx, r * math.cos(a), r * math.sin(a)])
    FI, FO, BI, BO = 0, n, 2 * n, 3 * n

    def quad(a, b, c, d):
        faces.append([a, b, c])
        faces.append([a, c, d])

    for k in range(n):
        k1 = (k + 1) % n
        quad(FI + k, FO + k, FO + k1, FI + k1)       # front (+x)
        quad(BI + k, BI + k1, BO + k1, BO + k)       # back (-x)
        quad(FO + k, BO + k, BO + k1, FO + k1)       # outer wall
        quad(FI + k, FI + k1, BI + k1, BI + k)       # inner wall
    return _add_mesh(spec, name, verts, faces)


def add_torus_mesh(spec, name: str, R: float, r: float, n_major: int = 48, n_minor: int = 12) -> str:
    """A torus about the local z axis (major radius R, tube radius r): a handwheel rim."""
    if _has(spec, "mesh", name):
        return name
    verts, faces = [], []
    for i in range(n_major):
        u = 2 * math.pi * i / n_major
        for j in range(n_minor):
            v = 2 * math.pi * j / n_minor
            rr = R + r * math.cos(v)
            verts.append([rr * math.cos(u), rr * math.sin(u), r * math.sin(v)])
    for i in range(n_major):
        i1 = (i + 1) % n_major
        for j in range(n_minor):
            j1 = (j + 1) % n_minor
            a, b, c, d = i * n_minor + j, i1 * n_minor + j, i1 * n_minor + j1, i * n_minor + j1
            faces.append([a, b, c])
            faces.append([a, c, d])
    return _add_mesh(spec, name, verts, faces)


# ----------------------------------------------------------------------------------------------- text drawing
_FONTS: dict = {}


def font(size_px: float):
    """Pillow's bundled scalable font (no system fonts needed) at `size_px` (cached)."""
    from PIL import ImageFont
    s = max(6, int(round(size_px)))
    f = _FONTS.get(s)
    if f is None:
        f = ImageFont.load_default(size=s)
        _FONTS[s] = f
    return f


def draw_centered_text(draw, cx: float, cy: float, text: str, fnt, fill, anchor="mm"):
    draw.text((cx, cy), text, font=fnt, fill=fill, anchor=anchor)


def fit_font(draw, text: str, max_w: float, max_h: float, start_px: float):
    s = int(start_px)
    while s > 7:
        f = font(s)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=f)
        if x1 - x0 <= max_w and y1 - y0 <= max_h:
            return f
        s -= 1
    return font(7)
