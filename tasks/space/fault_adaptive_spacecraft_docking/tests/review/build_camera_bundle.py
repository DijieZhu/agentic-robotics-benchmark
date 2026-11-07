"""Write the grader's camera bundle (a maintainer's tool; the grader never runs it).

    python3 -B build_camera_bundle.py --src <pr10_render>/bundle_v3 --out <task>/environment/assets/payload/render

The DOCK camera's scene (tests/camera.py) is drawn from the render-realism step's material bundle (2026-11-07,
`pr10_render/bundle_v3`, README there): the NASA Wind drum with its docking well, the boom and drogue, the NASA 2U
CubeSat with its probe and a thruster bell template, as meshes with texture coordinates; 13 PBR map sets at 512 px
(base sRGB, tangent normal, ORM); the fitted Filament look (per-material gains); the light rig (the sun, the image
light of the OVRTX probe of the scene's Earth, the skybox). This script copies the files camera.py reads into one
folder and writes its manifest, `bundle.json`: the light (the sun's direction and illuminance, the image light's
intensity; the agent camera's exposure is the camera's own, so the look's +0.5 EV for the demo is not applied), the
per-material gains, and the sha256 of every file, which camera.py checks before it builds the scene.

The bundle goes into the image root-only (environment/Dockerfile: /opt/fault-docking-render, mode 0700): with it and
camera.py an agent could render the grader's frames, a camera simulator; the agent gets the target's interface
document and sample frames instead (environment/setup/payload/given/). Until the owner approves the Hugging Face upload
the folder is staged locally at environment/assets/payload/render/ (gitignored) and the Dockerfile copies it.
"""

import argparse
import hashlib
import json
import os
import shutil

TEXTURES = "textures"          # the 512 px set (the look was fitted on it)
FORMAT = "fault_adaptive_spacecraft_docking camera bundle 1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="pr10_render/bundle_v3")
    ap.add_argument("--out", required=True, help="the bundle folder to write (emptied first)")
    a = ap.parse_args()
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "meshes"))
    os.makedirs(os.path.join(out, "textures"))
    rig = json.load(open(os.path.join(src, "rig_e1022", "rig.json")))
    mats = json.load(open(os.path.join(src, "materials.json")))
    meta = json.load(open(os.path.join(src, "meshes", "parts.json")))
    used = sorted({p["material"] for p in meta["parts"] if p.get("textured")})
    files = []

    def put(rel_src, rel_dst):
        shutil.copyfile(os.path.join(src, rel_src), os.path.join(out, rel_dst))
        files.append(rel_dst)

    put("meshes/parts.npz", "meshes/parts.npz")
    put("meshes/parts.json", "meshes/parts.json")
    put("materials.json", "materials.json")
    maps = sorted({mats[m]["maps"][k] for m in used for k in ("base", "normal", "orm")})
    for name in maps:
        put(os.path.join(TEXTURES, name), os.path.join("textures", name))
    put("rig_e1022/ibl.ktx", "ibl.ktx")
    put("rig_e1022/skybox.ktx", "skybox.ktx")
    manifest = {
        "format": FORMAT,
        "source": "pr10_render/bundle_v3 (render-realism step of the PR #10 difficulty round 2, 2026-11-07; look3.json "
                  "round r4): meshes with uvs, the 512 px maps, the e1022 v3 light rig (rig.json)",
        "textures": "textures (512 px)",
        "materials_used": used,
        "gains": {m: float(rig["gains"][m]) for m in sorted(rig["gains"]) if m in mats or m in meta["flat_materials"]},
        "sun_dir": [float(v) for v in rig["sun_dir"]],
        "sun_az_el_deg": [float(v) for v in rig["sun_az_el_deg"]],
        "sun_lux": float(rig["sun_lux"]),
        "ibl_intensity": float(rig["ibl"]),
        "note": "the look's ev_offset (+0.5 EV, the demo video's exposure) is not applied: the DOCK camera has its own "
                "fixed exposure (Filament's default camera, EV100 15)",
        "provenance": "meshes: NASA 3D Resources Wind.glb and 'CubeSat - 2 RU Generic.glb' (NASA media guidelines), cut "
                      "and scaled by pr10viz/parts.py; well, plate, boom, drogue, probe, bells procedural. Maps: "
                      "procedural and derived from the owner's A134 ISS pack (licensed for redistribution). Light: the "
                      "owner's High Resolution Earth 1.0 pack (Earth and star maps) through pr10viz/envmap3.py and an "
                      "OVRTX probe.",
        "files": {rel: sha256(os.path.join(out, rel)) for rel in sorted(files)},
    }
    with open(os.path.join(out, "bundle.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    total = sum(os.path.getsize(os.path.join(out, rel)) for rel in files)
    print("{} files, {:.1f} MB, bundle.json sha256 {}".format(len(files) + 1, total / 1e6,
                                                              sha256(os.path.join(out, "bundle.json"))))


if __name__ == "__main__":
    main()
