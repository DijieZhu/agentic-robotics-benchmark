"""Scanned PBR surface maps for a material of the Filament look (CC0 sets as published by Poly Haven: <name>_diff_<res>.jpg,
<name>_nor_gl_<res>.jpg (OpenGL normal), <name>_arm_<res>.jpg (R = ambient occlusion, G = roughness, B = metallic)).

The maps are cut to the geom's top face at their physical scale (no stretching: the set covers `map_m` metres), turned so
that the grain (along the image columns of the set) runs along the face's long side, and become the material's RGB,
NORMAL and ORM layers; the procedural texture of the scene stays the MuJoCo backend's.  Pixels only: no geometry moves.
"""
import glob
import hashlib
import os

import numpy as np

CACHE = {}


def map_files(directory):
    """{"diff", "nor_gl", "arm"} -> path of the one set in `directory`."""
    out = {}
    for kind in ("diff", "nor_gl", "arm"):
        hits = sorted(glob.glob(os.path.join(directory, "*_%s_*.jpg" % kind)) + glob.glob(os.path.join(directory, "*_%s_*.png" % kind)))
        if len(hits) != 1:
            raise FileNotFoundError("%s: need exactly one *_%s_* map, found %d" % (directory, kind, len(hits)))
        out[kind] = hits[0]
    return out


def set_digest(directory):
    h = hashlib.sha256()
    for kind, p in sorted(map_files(directory).items()):
        h.update(kind.encode())
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest()


def face_maps(directory, face_m, map_m):
    """uint8 (H, W, 3) diff / nor_gl / arm cut to a face of face_m = (x, y) metres; W runs along x.  Cached."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    key = (set_digest(directory), tuple(round(float(v), 6) for v in face_m), float(map_m))
    if key in CACHE:
        return CACHE[key]
    fx, fy = float(face_m[0]), float(face_m[1])
    out = {}
    for kind, p in map_files(directory).items():
        a = np.asarray(Image.open(p).convert("RGB"))
        ppm = a.shape[1] / float(map_m)
        along, across = int(round(max(fx, fy) * ppm)), int(round(min(fx, fy) * ppm))
        if along > a.shape[0] or across > a.shape[1]:
            raise ValueError("%s covers %.2f m; the face needs %.2f x %.2f m" % (p, map_m, fx, fy))
        a = a[:along, :across]                       # grain along the rows' direction (down the image)
        if fx >= fy:                                 # the face's long side is x (= texture width): transpose
            a = np.ascontiguousarray(a.transpose(1, 0, 2))
            if kind == "nor_gl":                     # a transpose swaps the tangent axes and flips both signs
                a = np.stack([255 - a[..., 1], 255 - a[..., 0], a[..., 2]], -1).astype(np.uint8)
        out[kind] = a
    CACHE[key] = out
    return out


def _albedo_gain(a, gain):
    """Scale an sRGB-encoded uint8 albedo by `gain` in linear light."""
    x = a.astype(np.float64) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * float(gain)
    lin = np.clip(lin, 0.0, 1.0)
    out = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (out * 255.0 + 0.5).astype(np.uint8)


def apply_surface(spec, material, geom, directory, map_m, roughness_scale=1.0, albedo_gain=1.0):
    """Give `material` (used by box geom `geom`) the scanned maps; returns a record."""
    import mujoco
    g = spec.geom(geom)
    if g is None:
        return None
    face = (2.0 * float(g.size[0]), 2.0 * float(g.size[1]))
    maps = face_maps(directory, face, map_m)
    if roughness_scale != 1.0:
        arm = maps["arm"].copy()
        arm[..., 1] = np.clip(arm[..., 1].astype(float) * float(roughness_scale), 0, 255).astype(np.uint8)
        maps = dict(maps, arm=arm)
    if albedo_gain != 1.0:
        maps = dict(maps, diff=_albedo_gain(maps["diff"], albedo_gain))
    roles = {"diff": (mujoco.mjtTextureRole.mjTEXROLE_RGB, mujoco.mjtColorSpace.mjCOLORSPACE_SRGB),
             "nor_gl": (mujoco.mjtTextureRole.mjTEXROLE_NORMAL, mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR),
             "arm": (mujoco.mjtTextureRole.mjTEXROLE_ORM, mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)}
    m = spec.material(material)
    for kind, (role, cs) in roles.items():
        a = maps[kind]
        name = "look_%s_%s" % (material, kind)
        t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=a.shape[1], height=a.shape[0], nchannel=3)
        t.colorspace = cs
        t.data = a.tobytes()
        m.textures[role] = name
    m.rgba = [1.0, 1.0, 1.0, 1.0]
    m.texrepeat = [1, 1]
    m.texuniform = False
    return {"material": material, "set": os.path.basename(os.path.normpath(directory)), "sha256": set_digest(directory),
            "face_m": face, "map_m": float(map_m), "px": list(maps["diff"].shape[:2]), "roughness_scale": float(roughness_scale),
            "albedo_gain": float(albedo_gain)}
