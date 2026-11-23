"""Review: a saved submission (a folder with run.sh) on development sites or on the graded sites, without a model
call. One fresh graded-mode service per site, as tests/grader.py starts it; no account switch and no network
filter, so run it as root in a throwaway container of the task's image. Text records only.

    /opt/venv/bin/python3 -B tests/review/replay_submission.py <submission dir> <out dir> --n 32
    /opt/venv/bin/python3 -B tests/review/replay_submission.py <submission dir> <out dir> --hidden

It separates a program's level from one draw of eight sites.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
KEYS = ("success", "target_outcome", "contact_wall", "contact_part", "target_ticks", "final_goal_distance",
        "min_duct_gap_m", "deck_support_feet", "duct_cleared", "deck_cleared", "max_forward_m")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args()
    graded = json.loads((TASK/"tests/hidden/sites.json").read_text())["seeds"]
    draws = [int(x) for x in np.random.default_rng(20261120).integers(1, 2**62, size=args.n)]
    seeds = graded if args.hidden else [s for s in draws if s not in set(graded)]
    client = Path("/opt/visual_robotics/client")
    if not client.exists():
        shutil.copytree(TASK/"tests/setup_payload", client)
    args.out.mkdir(parents=True, exist_ok=True)
    program = Path("/tmp/replay-program")
    shutil.copytree(args.submission, program, dirs_exist_ok=True)
    with (args.out/"results.jsonl").open("a") as results:
        for i, seed in enumerate(seeds):
            directory = args.out/"site{:02d}".format(i)
            directory.mkdir()
            (directory/"seed").write_text(str(seed))
            sock = "/tmp/replay-world.sock"
            if os.path.exists(sock):
                os.unlink(sock)
            service = subprocess.Popen(
                [sys.executable, "-B", str(TASK/"tests/runtime/service.py"), "--mode", "graded", "--directory",
                 str(directory), "--socket", sock, "--seed-file", str(directory/"seed")],
                cwd=str(TASK/"tests/runtime"), stdout=open(directory/"service.log", "w"), stderr=subprocess.STDOUT)
            started = time.monotonic()
            while not (directory/"ready").exists():
                time.sleep(.05)
            home = Path("/tmp/replay-home{}".format(i))
            (home/"tmp").mkdir(parents=True, exist_ok=True)
            env = {"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(home/"tmp"),
                   "LANG": "C.UTF-8", "ROBOT_SOCKET": sock, "MUJOCO_GL": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
            try:
                code = subprocess.run(["/bin/bash", "run.sh"], cwd=str(program), env=env, timeout=1800,
                                      stdout=open(directory/"program.log", "w"), stderr=subprocess.STDOUT).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            service.terminate()
            service.wait(timeout=10)
            metrics = json.loads((directory/"state.json").read_text())["metrics"]
            (directory/"seed").unlink()
            row = {"site": i, "exit": code, "wall_s": round(time.monotonic()-started, 1),
                   **{key: metrics.get(key) for key in KEYS}}
            print(json.dumps(row), flush=True)
            results.write(json.dumps(row)+"\n")
            results.flush()


if __name__ == "__main__":
    main()
