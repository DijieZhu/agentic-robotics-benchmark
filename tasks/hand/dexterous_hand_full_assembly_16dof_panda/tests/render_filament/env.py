"""An equirectangular environment (an HDRI as a float32 .npy, linear RGB, row 0 = up) -> a MuJoCo cube texture (skybox) for
the Filament look: the background of every photo and, through an image light, the scene's ambient light and reflections.

MuJoCo cube data = 6 faces stacked vertically (width N, height 6 N), measured on the Filament skybox (cube_probe):
face 0 = +x, 1 = -x, 2 = +z (up), 3 = -z (down), 4 = -y, 5 = +y; texel (row i, col j) of face k looks along
    d = F[k] + a R[k] + b U[k],   a = (2 j + 1) / N - 1,   b = 1 - (2 i + 1) / N.
The HDRI is mapped with z up: elevation = asin(d_z), azimuth = atan2(d_y, d_x) + yaw; tone curve: middle grey of the
environment's median luminance x 2^ev -> 0.18, an optional partial white balance, clipped, sRGB-encoded (MuJoCo textures
are 8-bit).
"""
import hashlib
import math
import os

import numpy as np

FACES = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
         ((0, 0, -1), (1, 0, 0), (0, -1, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 1, 0), (-1, 0, 0), (0, 0, 1))]
CACHE = {}


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cube_faces(path, n=1024, yaw_deg=0.0, ev=0.0, wb=0.0):
    """uint8 (6 n, n, 3) cube data of the equirect HDRI at `path` (cached per file digest and parameters).  `wb` in [0, 1]
    moves the environment's median colour that far towards neutral grey (a partial white balance)."""
    key = (file_digest(path), n, float(yaw_deg), float(ev), float(wb))
    if key in CACHE:
        return CACHE[key]
    eq = np.load(path).astype(np.float32)
    h, w = eq.shape[:2]
    lum = 0.2126 * eq[..., 0] + 0.7152 * eq[..., 1] + 0.0722 * eq[..., 2]
    k = 0.18 / max(1e-6, float(np.median(lum))) * (2.0 ** ev)
    med = np.median(eq.reshape(-1, 3), axis=0)
    gain = (float(med.mean()) / np.maximum(med, 1e-6)) ** float(wb)
    t = (np.arange(n) * 2 + 1) / n - 1.0
    a, b = np.meshgrid(t, -t)                     # a along columns, b = up for row 0
    out = []
    for F, R, U in FACES:
        d = np.asarray(F, float)[None, None, :] + a[..., None] * np.asarray(R, float) + b[..., None] * np.asarray(U, float)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        az = np.arctan2(d[..., 1], d[..., 0]) + math.radians(yaw_deg)
        el = np.arcsin(np.clip(d[..., 2], -1, 1))
        u = ((0.5 - az / (2 * math.pi)) % 1.0) * w
        v = (0.5 - el / math.pi) * h
        u0, v0 = np.floor(u).astype(int), np.clip(np.floor(v).astype(int), 0, h - 1)
        fu, fv = (u - u0)[..., None], (v - v0)[..., None]
        u1, v1 = (u0 + 1) % w, np.clip(v0 + 1, 0, h - 1)
        u0 %= w
        c = (eq[v0, u0] * (1 - fu) + eq[v0, u1] * fu) * (1 - fv) + (eq[v1, u0] * (1 - fu) + eq[v1, u1] * fu) * fv
        out.append((_srgb(c * k * gain) * 255.0 + 0.5).astype(np.uint8))
    cube = np.concatenate(out, 0)
    CACHE[key] = cube
    return cube


def add_environment(spec, path, intensity, n=1024, yaw_deg=0.0, ev=0.0, wb=0.0, ibl_path=None, ibl_n=256, image_light=True):
    """Replace the spec's skybox with the environment and light the scene with it (an image light).  With `ibl_path`
    (a light probe in the bench frame, e.g. a room's baked reflection probe) the image light uses that probe instead and
    the environment is only the background (a separate CUBE texture: MuJoCo's Filament takes any cube for the light)."""
    import mujoco
    for t in list(spec.textures):
        if int(t.type) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            spec.delete(t)
    cube = cube_faces(path, n=n, yaw_deg=yaw_deg, ev=ev, wb=wb)
    t = spec.add_texture(name="look_environment", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, width=n, height=6 * n, nchannel=3)
    t.data = cube.tobytes()
    light_tex = "look_environment"
    rec = {"file": os.path.basename(path), "sha256": file_digest(path), "face_px": n, "yaw_deg": yaw_deg, "ev": ev, "wb": wb,
           "intensity": float(intensity)}
    if ibl_path:
        pc = cube_faces(ibl_path, n=ibl_n, yaw_deg=0.0, ev=0.0, wb=wb)
        p = spec.add_texture(name="look_probe", type=mujoco.mjtTexture.mjTEXTURE_CUBE, width=ibl_n, height=6 * ibl_n, nchannel=3)
        p.data = pc.tobytes()
        light_tex = "look_probe"
        rec["probe"] = {"file": os.path.basename(ibl_path), "sha256": file_digest(ibl_path), "face_px": ibl_n}
    if image_light:                                  # False: the caller adds an HDR (KTX) image light itself
        spec.worldbody.add_light(name="look_environment", type=mujoco.mjtLightType.mjLIGHT_IMAGE, texture=light_tex,
                                 intensity=float(intensity))
    rec["image_light"] = bool(image_light)
    return rec
