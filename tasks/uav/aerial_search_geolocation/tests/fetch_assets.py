"""Download the build-time assets of the task images. Runs during `docker build` only.

    fetch_assets.py scenes --repo R --revision V --maps A B --out /opt/aerialsim/scenes
    fetch_assets.py spawn-areas --url U --sha256 H --out /tmp/map_spawnarea_info.json
    fetch_assets.py models --out /opt/models

`scenes` fetches TravelUAV's packaged Unreal scenes (AirSim plugin built in) from the public,
ungated Hugging Face dataset R at revision V: the `closeloop_envs` archive, split in five
parts, of which only the named maps, their launch scripts and the shared Engine folder are
extracted. `spawn-areas` fetches AeroVLA's per-map list of object placement areas and checks
its hash. `models` fetches the two open-vocabulary models a controller may use offline.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import urllib.request

MODELS = ("openai/clip-vit-base-patch16", "google/owlv2-base-patch16-ensemble")
PARTS = ("closeloop_envs.z01", "closeloop_envs.z02", "closeloop_envs.z03",
         "closeloop_envs.z04", "closeloop_envs.zip")
SCRIPT = {"ModernCityMap": "ModernCityMap"}          # map name -> launch script stem
FOLDER = {"ModernCityMap": "ModernCityDowntown"}     # map name -> content folder


def scenes(args):
    # Imported here: `spawn-areas` runs on the base image's python3, which lacks it.
    from huggingface_hub import hf_hub_download
    work = os.path.join(args.out, ".download")
    os.makedirs(work, exist_ok=True)
    for part in PARTS:
        hf_hub_download(args.repo, part, repo_type="dataset", revision=args.revision,
                        local_dir=work)
    wanted = ["closeloop_envs/Engine/*", "closeloop_envs/Manifest_NonUFSFiles_Linux.txt"]
    for name in args.maps:
        wanted += ["closeloop_envs/{}.sh".format(SCRIPT.get(name, name)),
                   "closeloop_envs/{}/*".format(FOLDER.get(name, name))]
    subprocess.run(["7z", "x", "-y", "-o" + args.out, os.path.join(work, "closeloop_envs.zip")]
                   + wanted + ["-r"], check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL)
    shutil.rmtree(work)
    root = os.path.join(args.out, "closeloop_envs")
    for name in args.maps:
        script = os.path.join(root, SCRIPT.get(name, name) + ".sh")
        binary = os.path.join(root, FOLDER.get(name, name), "Binaries", "Linux",
                              FOLDER.get(name, name))
        if not (os.path.isfile(script) and os.path.isfile(binary)):
            raise SystemExit("scene {} incomplete after extraction".format(name))
        os.chmod(script, 0o755)
        os.chmod(binary, 0o755)


def spawn_areas(args):
    with urllib.request.urlopen(args.url) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.sha256:
        raise SystemExit("spawn-area file hash {} != expected {}".format(digest, args.sha256))
    with open(args.out, "wb") as handle:
        handle.write(data)


def models(args):
    from huggingface_hub import snapshot_download
    for repo in MODELS:
        snapshot_download(repo, local_dir=os.path.join(args.out, repo.split("/")[1]),
                          allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "*.txt",
                                          "*.model"])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="what", required=True)
    s = sub.add_parser("scenes")
    s.add_argument("--repo", required=True)
    s.add_argument("--revision", required=True)
    s.add_argument("--maps", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = sub.add_parser("spawn-areas")
    a.add_argument("--url", required=True)
    a.add_argument("--sha256", required=True)
    a.add_argument("--out", required=True)
    m = sub.add_parser("models")
    m.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scenes": scenes, "spawn-areas": spawn_areas, "models": models}[args.what](args)


if __name__ == "__main__":
    main()
