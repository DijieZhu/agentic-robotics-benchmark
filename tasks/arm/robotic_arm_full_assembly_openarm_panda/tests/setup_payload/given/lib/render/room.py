"""Room bundles: a static environment (floor, walls, furniture, lamps) around the bench of a photo scene.

A bundle (format arb-room/1) is a directory built at authoring time from an art asset pack (arb-work unity_tools:
ArbExport.cs + build_room.py): room.json, meshes/<material>.npz (v, n, uv, f; the bench frame: z up, bench top z = 0,
one mesh per material), textures/*.png (rgb, normal, orm = ambient occlusion / roughness / metallic).

    add_room(spec, bundle) -> record          one textured mesh geom per material (static, no contacts, geom group 2,
                                              names room_*) and the room's lamps (spot lights room_light_*)
    resolve(name_or_path) -> directory        a bundle name is looked up under $ARB_ROOM_DIR

Geometry only: the Filament look decides how bright the lamps are (render_filament.backend.Look.room_*).  No room object
stands in the space the photo protocol's cameras use (build_room.py clears CAMERA_BOX).
"""
import hashlib
import json
import math
import os

import numpy as np

FORMAT = "arb-room/1"
SHADOW_LAMPS_R = 1.6        # lamps within this horizontal distance of the bench centre cast shadows
GLASS_RGBA = [0.92, 0.95, 0.97, 0.28]
BAKED_EMISSION = 1.0        # emission of a baked material's texture (1.0 = room.json baked.scale); the look rescales it
BAKED_MIN_ROUGHNESS = 0.3   # the live specular of a baked surface is lit by one probe: keep it from turning into a mirror


def resolve(name_or_path):
    if os.path.isdir(name_or_path):
        return os.path.abspath(name_or_path)
    base = os.environ.get("ARB_ROOM_DIR")
    if not base:
        raise FileNotFoundError("room %r: not a directory and $ARB_ROOM_DIR is not set" % name_or_path)
    p = os.path.join(base, name_or_path)
    if not os.path.isdir(p):
        raise FileNotFoundError("room %r not under %s" % (name_or_path, base))
    return p


def digest(bundle):
    """sha256 over room.json and every mesh / texture file of the bundle."""
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(bundle)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, bundle).encode())
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
    return h.hexdigest()


def _texture(spec, name, path, colorspace, alpha):
    import mujoco
    t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
    t.colorspace = colorspace
    t.nchannel = 4 if alpha else 3
    return name


def add_room(spec, bundle, bench_xy=(0.0, 0.15)):
    import mujoco
    bundle = resolve(bundle)
    room = json.load(open(os.path.join(bundle, "room.json")))
    if room.get("format") != FORMAT:
        raise ValueError("%s: format %r, expected %r" % (bundle, room.get("format"), FORMAT))
    srgb, lin = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB, mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    R = mujoco.mjtTextureRole
    nfaces = 0
    for me in room["meshes"]:
        m = room["materials"][me["material"]]
        tex = m.get("textures", {})
        alpha = bool(m.get("transparent"))
        if m.get("baked"):                            # a baked bundle (render_cycles/bake_room.py)
            # the lightmap is the DIFFUSE radiance (emission, not lit again); the view-dependent specular stays live:
            # a black base colour with the surface's own roughness (dielectric F0 4 %), or for a metal its colour as F0
            mat = spec.add_material(name="room_baked_" + me["name"])
            # effective terms (bake_room / the bundle: mean of the ORM map where there is one; HDRP's scalar metallic
            # is only the remap ceiling of the mask map)
            metal = float(m.get("metallic_eff", m.get("metallic", 0.0))) >= 0.5
            mat.rgba = [float(x) for x in m["rgba"][:3]] + [1.0] if metal else [0.0, 0.0, 0.0, 1.0]
            mat.roughness = float(np.clip(m.get("roughness_eff", m.get("roughness", 0.6)), BAKED_MIN_ROUGHNESS, 1.0))
            mat.metallic = 1.0 if metal else 0.0
            mat.specular, mat.reflectance = 0.5, 0.0
            mat.emission = BAKED_EMISSION              # the Filament look rescales it (Look.room_baked_emission)
            mat.textures[R.mjTEXROLE_EMISSIVE] = _texture(spec, "room_tex_%s_baked" % me["name"],
                                                          os.path.join(bundle, "textures", m["baked"]["file"]), srgb, False)
            tex = {}
            alpha = False
        else:
            mat = spec.add_material(name="room_mat_" + me["name"])
        if not m.get("baked"):
            rgba = [float(x) for x in m["rgba"]]
            # HDRP glass is transparent with base alpha 0 (specular only); Filament's transparent PBR fades it to nothing
            # or black: a light frosted glass instead (GLASS_RGBA)
            rgba = GLASS_RGBA if alpha else rgba[:3] + [1.0]
            mat.rgba = rgba
            mat.roughness = 1.0 if "orm" in tex else float(np.clip(m["roughness"], 0.02, 1.0))   # x the map if any
            mat.metallic = 1.0 if "orm" in tex else float(np.clip(m["metallic"], 0.0, 1.0))
            em = float(max(m.get("emissive_rgb") or [0.0]))
            if em > 0:
                mat.emission = float(min(4.0, 1.0 + math.log10(1.0 + em)))
        has_orm = "orm" in tex
        if "rgb" in tex and not alpha:                # glass: no base map (HDRP's is a dark refraction tint)
            mat.textures[R.mjTEXROLE_RGB] = _texture(spec, "room_tex_%s_rgb" % me["name"],
                                                     os.path.join(bundle, "textures", tex["rgb"]), srgb, False)
        if "normal" in tex:
            mat.textures[R.mjTEXROLE_NORMAL] = _texture(spec, "room_tex_%s_nrm" % me["name"],
                                                        os.path.join(bundle, "textures", tex["normal"]), lin, False)
        if "emissive" in tex:                         # the lit part of a lamp (Unity's emission map)
            mat.textures[R.mjTEXROLE_EMISSIVE] = _texture(spec, "room_tex_%s_emi" % me["name"],
                                                          os.path.join(bundle, "textures", tex["emissive"]), srgb, False)
        if has_orm:
            mat.textures[R.mjTEXROLE_ORM] = _texture(spec, "room_tex_%s_orm" % me["name"],
                                                     os.path.join(bundle, "textures", tex["orm"]), lin, False)
        z = np.load(os.path.join(bundle, me["file"]))
        uv = np.asarray(z["uv"], np.float64).copy()
        uv[:, 1] = 1.0 - uv[:, 1]                  # Unity: v = 0 at the bottom row; MuJoCo: at the top row
        mesh = spec.add_mesh(name="room_mesh_" + me["name"])
        mesh.uservert = np.asarray(z["v"], np.float64).ravel()
        mesh.usernormal = np.asarray(z["n"], np.float64).ravel()
        mesh.usertexcoord = uv.ravel()
        mesh.userface = np.asarray(z["f"], np.int32).ravel()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        g = spec.worldbody.add_geom(name="room_" + me["name"], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                                    material=mat.name)
        g.contype = 0
        g.conaffinity = 0
        g.group = 2
        nfaces += int(len(z["f"]))
    lamps = 0
    for k, l in enumerate(room.get("lights", [])):
        if not l.get("active", True) or l.get("type") == "Directional":
            continue
        p = np.asarray(l["position"], float)
        f = np.asarray(l["forward"], float)
        near = math.hypot(p[0] - bench_xy[0], p[1] - bench_xy[1]) <= SHADOW_LAMPS_R
        li = spec.worldbody.add_light(name="room_light_%d" % k, type=mujoco.mjtLightType.mjLIGHT_SPOT)
        li.pos = [float(x) for x in p]
        li.dir = [float(x) for x in f / max(1e-9, np.linalg.norm(f))]
        li.cutoff = 80.0
        li.exponent = 1.0
        li.castshadow = bool(near)
        li.diffuse = [float(x) for x in l["color"]]
        li.intensity = float(l["intensity"])        # source units; the look scales them
        li.bulbradius = float(max(l.get("area_size", [0.05, 0.05]))) * 0.5
        lamps += 1
    return {"room": os.path.basename(bundle), "format": FORMAT, "meshes": len(room["meshes"]), "faces": nfaces,
            "lamps": lamps, "provenance": room.get("provenance")}
