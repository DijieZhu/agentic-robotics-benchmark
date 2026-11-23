"""MuJoCo (MjSpec) integration: a parcel's visual mesh and maps as a non-colliding, massless visual geom.

    import mujoco
    from parcelkit import make_parcel
    from parcelkit.mj import add_parcel_visual
    rec = make_parcel(parcel, seed, ("agent",), out_dir)
    body = spec.worldbody.add_body(name="parcel3")            # the simulator's parcel body (collision geoms as now)
    add_parcel_visual(spec, body, rec, "agent", prefix="parcel3")

Textures: base colour as mjTEXROLE_RGB (sRGB), the normal map as mjTEXROLE_NORMAL and the ORM map as mjTEXROLE_ORM
(both linear). MuJoCo's classic renderer uses the RGB texture only; the Filament renderer reads all three.
UVs: MuJoCo's texture rows run top-down with v = 0 at the image's first row, so the kit's (USD / OBJ) v is flipped
here (``V_FLIP``); checked by rendering a waybill of known direction.
"""
from __future__ import annotations

import os

import numpy as np

V_FLIP = True


def _tex_arrays(rec, quality, base_dir=None):
    arr = rec.get("_arrays", {}).get(quality)
    if arr is not None:
        return arr["V"], arr["N"], arr["UV"], arr["F"], arr["tex"]
    # from files written by make_parcel
    from PIL import Image
    files = rec["outputs"][quality]["files"]
    d = base_dir or "."
    tex = {k: np.asarray(Image.open(os.path.join(d, files[k])).convert("RGB")) for k in ("basecolor", "normal", "orm")}
    V, N, UV, F = read_obj(os.path.join(d, files["obj"]))
    return V, N, UV, F, tex


def read_obj(path):
    V, T, Nn, F = [], [], [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            T.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("vn "):
            Nn.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(Nn), np.array(T), np.array(F, int)


def add_parcel_visual(spec, body, rec, quality="agent", prefix=None, group=0, base_dir=None, normal_orm=True):
    """Add the parcel's textures, material, mesh and one visual geom (contype 0, conaffinity 0, mass 0) to ``body``.
    Returns the geom (mjsGeom). ``rec``: a make_parcel record (with arrays) or a loaded record JSON (files read from
    ``base_dir``)."""
    import mujoco
    prefix = prefix or rec["name"]
    V, N, UV, F, tex = _tex_arrays(rec, quality, base_dir)

    def add_tex(name, img, srgb):
        t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=int(img.shape[1]),
                             height=int(img.shape[0]), nchannel=3)
        t.data = np.ascontiguousarray(img[..., :3], np.uint8).tobytes()
        t.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB if srgb else mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
        return name
    mat = spec.add_material(name=prefix + "_look")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = add_tex(prefix + "_rgb", tex["basecolor"], True)
    if normal_orm:
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_NORMAL] = add_tex(prefix + "_nrm", tex["normal"], False)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_ORM] = add_tex(prefix + "_orm", tex["orm"], False)
    mat.rgba = [1.0, 1.0, 1.0, 1.0]
    mat.specular = 0.3
    mat.shininess = 0.1
    mat.reflectance = 0.0
    me = spec.add_mesh(name=prefix + "_mesh")
    me.uservert = np.asarray(V, np.float64).ravel().tolist()
    me.usernormal = np.asarray(N, np.float64).ravel().tolist()
    uv = np.asarray(UV, np.float64).copy()
    if V_FLIP:
        uv[:, 1] = 1.0 - uv[:, 1]
    me.usertexcoord = uv.ravel().tolist()
    me.userface = np.asarray(F, np.int32).ravel().tolist()
    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    g = body.add_geom(name=prefix + "_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=me.name, contype=0,
                      conaffinity=0, density=0.0, mass=0.0, group=group, material=mat.name)
    g.rgba = [1.0, 1.0, 1.0, 1.0]
    return g


def add_parcel_collision(body, kind, size, mass, friction, prefix, group=3, rgba=(0.9, 0.3, 0.2, 0.0)):
    """The simulator's collision geoms for one parcel (parcelcell._geom.parcel_fields), for tests and proofs."""
    import mujoco
    from parcelcell import _geom
    f = _geom.parcel_fields(kind, size, mass, friction)
    body.add_geom(name=prefix + "_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(map(float, f["box_size"])),
                  friction=f["box_friction"], solimp=f["box_solimp"], solref=f["box_solref"], group=group,
                  rgba=list(rgba), mass=float(mass) * (0.9 if kind == "soft" else 1.0))
    if kind == "soft":
        body.add_geom(name=prefix + "_pad", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      size=list(map(float, f["pad_size"])), friction=f["pad_friction"], solimp=f["pad_solimp"],
                      solref=f["pad_solref"], group=group, rgba=list(rgba), mass=float(mass) * 0.1)
