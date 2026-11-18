"""Run one submission folder on one site over the real wire, the way tests/verify.py runs a mission (a fresh cell in
this process, the program as a child process through the client's launcher, JSON lines on its stdin / stdout,
cellkit.wire's program-time and silence caps), without the verify stage's lockdown, account switch and network
filter. For the maintainers: the reference check, the baselines, the determinism test.

    MUJOCO_GL=osmesa python -I tests/review/cell/run_wire.py --site SITE --program DIR --out OUT_DIR \
        [--privileged] [--seed N] [--sim-time-limit S] [--name NAME]

OUT_DIR gets summary.json (the cell's finalize() plus the wire's end reason and timings), debrief.json, traj.npz and
program.log (the program's stderr). The process exits 0 when the mission ran (whatever it scored).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.dirname(TESTS)
sys.path.insert(0, TESTS)

from cellkit import runner, wire  # noqa: E402
from plantcell.client import launcher_argv  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402
from plantcell.wire_guard import GuardedProcess  # noqa: E402

LIB = os.path.join(TASK, "environment", "setup", "payload", "given", "lib")


def grader_config():
    p = os.path.join(TESTS, "grader_config.json")
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"program_time_cap_s": 1200.0, "privileged_program_time_cap_s": 3600.0, "silence_s": 180.0,
                "cpu_cap_factor": 4.0}


def run(site, program, out, privileged=False, seed=None, sim_time_limit=None, name=None, log=print,
        program_cap=None):
    cfg = grader_config()
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    server = PlantCellServer(config=cfg, mode="hidden", privileged=privileged, log=log,
                             reference_dir=os.path.join(TESTS, "reference") if privileged else None)
    server.reset(site, seed=seed, kind="hidden", sim_time_limit=sim_time_limit, name=name)
    build_s = time.time() - t0
    run_dir = tempfile.mkdtemp(prefix="plantcell_run_")
    prog = os.path.join(run_dir, "program")
    shutil.copytree(program, prog, ignore=shutil.ignore_patterns("__pycache__"))
    home, tmp = os.path.join(run_dir, "home"), os.path.join(run_dir, "tmp")
    os.makedirs(home)
    os.makedirs(tmp)
    # the program's environment as tests/verify.py builds it (cellkit.runner.child_env plus one rasteriser thread
    # pair): nothing of this process's environment except PATH and MUJOCO_GL
    env = runner.child_env(home, tmp, LIB, os.path.dirname(LIB))
    env["LP_NUM_THREADS"] = "2"
    cap = float(program_cap or (cfg.get("privileged_program_time_cap_s", 3600.0) if privileged
                                else cfg.get("program_time_cap_s", 1200.0)))
    cpu_s = float(cfg.get("cpu_cap_factor", 4.0)) * float(cfg.get("program_time_cap_s", 1200.0))
    argv = launcher_argv(sys.executable, LIB, prog, "patrol")
    t1 = time.time()
    proc = GuardedProcess(argv, cwd=prog, env=env, stderr_path=os.path.join(out, "program.log"),
                          cpu_s=None if privileged else cpu_s)      # as verify.py (the request-line guard)
    try:
        reason = wire.serve_episode(server, proc, cap, float(cfg.get("silence_s", 180.0)), log=log)
    finally:
        proc.terminate()
    if not server.done:
        server._end(reason)
    summary = server.finalize()
    summary["end_reason"] = reason
    summary["episode_wall_s"] = round(time.time() - t1, 2)
    summary["cell_build_s"] = round(build_s, 2)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    with open(os.path.join(out, "debrief.json"), "w") as f:
        json.dump(server.debrief(true_pose_hz=2.0), f, indent=1)
    if isinstance(server.reference_result, dict):           # the privileged reference's own log and crossings
        with open(os.path.join(out, "reference_log.json"), "w") as f:
            json.dump(server.reference_result, f, indent=1, default=str)
    import numpy as np
    np.savez_compressed(os.path.join(out, "traj.npz"), traj=server.trajectory(),
                        columns=np.array(summary["trajectory_columns"]))
    server.close()
    shutil.rmtree(run_dir, ignore_errors=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--privileged", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sim-time-limit", type=float, default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    s = run(a.site, a.program, a.out, a.privileged, a.seed, a.sim_time_limit, a.name,
            log=lambda m: print(m, file=sys.stderr, flush=True))
    keep = ("episode_score", "R", "parts", "done_reason", "end_reason", "sim_time_s", "episode_wall_s",
            "program_time_s", "cell_work_s", "physics_wall_s", "render_wall_s", "frames", "reports_made",
            "reports_delivered", "dose_fraction", "distance_m", "illegal_call_count", "internal_errors",
            "final_state_sha256", "reference")
    print(json.dumps({k: s.get(k) for k in keep}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
