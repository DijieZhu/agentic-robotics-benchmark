"""Fetch licensed, pinned nominal robot assets; no scene instances or answers."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def bake_models(manifest, destination):
    """Convert pinned glTF scenes to MuJoCo OBJ/PBR assets without changing scale."""
    import numpy as np
    import trimesh
    for item in manifest.get("pbr_models", []):
        folder = destination/"baked"/item["name"]
        folder.mkdir(parents=True, exist_ok=True)
        scene = trimesh.load(destination/item["gltf"], force="scene", process=False)
        meshes = []
        # glTF may reuse node names; trimesh disambiguates them with random UUIDs.
        # Sort by actual geometry/transform so independent image builds agree.
        def node_key(node):
            transform, geometry = scene.graph[node]
            mesh = scene.geometry[geometry]
            data = transform.tobytes()+mesh.vertices.tobytes()+mesh.faces.tobytes()
            if mesh.visual.uv is not None:
                data += mesh.visual.uv.tobytes()
            return hashlib.sha256(data).hexdigest()

        for node in sorted(scene.graph.nodes_geometry, key=node_key):
            transform, geometry = scene.graph[node]
            mesh = scene.geometry[geometry].copy()
            mesh.apply_transform(transform)
            # glTF is Y-up; MuJoCo is Z-up.
            mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                           [0, 1, 0, 0], [0, 0, 0, 1]]))
            meshes.append(mesh)
        bounds = np.array([np.vstack([m.bounds for m in meshes])[::2].min(0),
                           np.vstack([m.bounds for m in meshes])[1::2].max(0)])
        offset = np.r_[(bounds[0, :2]+bounds[1, :2])/2, bounds[0, 2]]
        parts = []
        for i, mesh in enumerate(meshes):
            mesh.apply_translation(-offset)
            material = mesh.visual.material
            record = {"mesh": f"part{i}.obj", "textures": {},
                      "rgba": ([1., 1., 1., 1.] if material.baseColorFactor is None
                               else (np.asarray(material.baseColorFactor)/255.).tolist()),
                      "roughness": float(1. if material.roughnessFactor is None else material.roughnessFactor),
                      "metallic": float(1. if material.metallicFactor is None else material.metallicFactor)}
            for role, attribute in (("rgb", "baseColorTexture"), ("normal", "normalTexture"),
                                     ("orm", "metallicRoughnessTexture")):
                image = getattr(material, attribute, None)
                if image is not None:
                    image = image.convert("RGB")
                    image.thumbnail((1024, 1024))
                    filename = f"part{i}_{role}.png"
                    image.save(folder/filename)
                    record["textures"][role] = filename
            # Trimesh already converts glTF UV origins on import. Preserve its
            # OBJ coordinates; another flip misaligns scanned texture atlases.
            (folder/record["mesh"]).write_text(trimesh.exchange.obj.export_obj(
                mesh, include_normals=True, include_texture=True, write_texture=False))
            parts.append(record)
        (folder/"model.json").write_text(json.dumps({"size": (bounds[1]-bounds[0]).tolist(),
                                                   "parts": parts}, indent=2)+"\n")


def fetch(manifest, destination):
    manifest = json.loads(Path(manifest).read_text())
    destination = Path(destination)
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe asset path")
        target = destination/relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
            continue
        url = item.get("url") or f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{manifest["revision"]}/{relative.as_posix()}'
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"asset checksum mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    bake_models(manifest, destination)
    (destination/"provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    fetch(args.manifest, args.destination)
