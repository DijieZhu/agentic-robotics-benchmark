"""Unpack TravelUAV scenes into a task image. Runs during `docker build` only.

    unpack_scenes.py --archive DIR --maps A B --out /opt/aerialsim/scenes

DIR holds TravelUAV's packaged Unreal scenes (AirSim plugin built in): the `closeloop_envs`
archive of the public Hugging Face dataset wangxiangyu0814/TravelUAV_env at revision
44de5739a95a2f6a88767446421cddada9606642, split in five parts, as the image fetched them from
the benchmark's asset dataset (assets.sha256). Only the named maps, their launch scripts and
the shared Engine folder are extracted.
"""

import argparse
import os
import subprocess

SCRIPT = {"ModernCityMap": "ModernCityMap"}          # map name -> launch script stem
FOLDER = {"ModernCityMap": "ModernCityDowntown"}     # map name -> content folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wanted = ["closeloop_envs/Engine/*", "closeloop_envs/Manifest_NonUFSFiles_Linux.txt"]
    for name in args.maps:
        wanted += ["closeloop_envs/{}.sh".format(SCRIPT.get(name, name)),
                   "closeloop_envs/{}/*".format(FOLDER.get(name, name))]
    subprocess.run(["7z", "x", "-y", "-o" + args.out, os.path.join(args.archive, "closeloop_envs.zip")]
                   + wanted + ["-r"], check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL)
    root = os.path.join(args.out, "closeloop_envs")
    for name in args.maps:
        script = os.path.join(root, SCRIPT.get(name, name) + ".sh")
        binary = os.path.join(root, FOLDER.get(name, name), "Binaries", "Linux",
                              FOLDER.get(name, name))
        if not (os.path.isfile(script) and os.path.isfile(binary)):
            raise SystemExit("scene {} incomplete after extraction".format(name))
        os.chmod(script, 0o755)
        os.chmod(binary, 0o755)


if __name__ == "__main__":
    main()
